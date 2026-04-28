# Copyright (C) 2026 Rikiza89
# Licensed under the Apache License, Version 2.0
"""
BILT_Workflow_JA バックエンドサービス (port 5002)
ハンドルする機能:
  - BILT モデルトレーニング（事前学習モデルなし、ゼロから学習）
  - BILT 物体検知（カメラフレーム推論）
  - カメラ管理

起動方法: python bilt_service.py
"""

import os
import sys
import json
import socket
import base64
import threading
import time
import logging
import traceback
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request, jsonify, Response
from PIL import Image
from werkzeug.utils import secure_filename

# ── パス初期設定 ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# bilt パッケージを追加
BILT_REPO_DIR = os.path.join(os.path.dirname(BASE_DIR), 'bilt')
if os.path.isdir(BILT_REPO_DIR) and BILT_REPO_DIR not in sys.path:
    sys.path.insert(0, BILT_REPO_DIR)

from config import config as app_config, Config
from bilt_managers import (EnhancedCameraManager, DetectionProcessor, RGBBalancer,
                           ChainDetectionManager, ImageManager)

# ── Flask アプリ ─────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config.from_object(app_config['default'])
Config.create_directories()

logging.basicConfig(
    level=getattr(logging, app.config.get('LOG_LEVEL', 'INFO')),
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler('bilt_service.log'), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

BILT_MODELS_DIR = app.config.get('BILT_MODELS_DIR', os.path.join(BASE_DIR, 'bilt_models'))
os.makedirs(BILT_MODELS_DIR, exist_ok=True)

# ── グローバル変数 ───────────────────────────────────────────────────────────────
current_bilt_model = None        # BILT Inferencer instance
current_bilt_model_name = None   # loaded .pth filename
model_info = {'name': None, 'classes': [], 'loaded': False, 'variant': None}

latest_detections = []
detection_lock = threading.Lock()

camera_manager = EnhancedCameraManager()
rgb_balancer = RGBBalancer()
# camera_manager.release() clears current_index, so we keep a separate copy
# that persists across stop/start cycles.
_bilt_selected_camera_index = None

detection_thread = None
detection_active = False
frame_lock = threading.Lock()
latest_frame = None
latest_frame_size = (640, 480)   # (w, h) — updated each frame

# Capture thread — drains the camera buffer at full speed independently of
# inference so the MJPEG stream is always smooth and inference always gets
# the freshest available frame.
_capture_frame  = None
_capture_lock   = threading.Lock()
_capture_active = False
_capture_thread = None

detection_settings = {
    'conf': 0.80,
    'iou': 0.45,
    'max_det': 5,
    'classes': None,
    'counter_mode': False,
    'dataset_capture': False,
    'project_folder': '',
    'chain_mode': False,
    'chain_steps': [],
    'chain_timeout': 5.0,
    'chain_auto_advance': True,
    'chain_pause_time': 10.0,
}

object_counters = {}
chain_state = ChainDetectionManager.make_initial_state()

detection_stats = {'total_detections': 0, 'fps': 0, 'last_detection_time': None}

# Training state
training_active = False
training_thread = None
training_state = {
    'active': False,
    'epoch': 0,
    'total_epochs': 0,
    'loss': None,
    'val_loss': None,
    'metrics': {},
    'log_lines': [],
    'epoch_history': [],
    'phase': 'idle',
    'error': None,
    'model_path': None,
    'training_time': None,
    'best_val_loss': None,
}
training_state_lock = threading.Lock()
_MAX_LOG_LINES = 500


# ─────────────────────────────────────────────────────────────────────────────
# ヘルスチェック
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'service': 'bilt'})


# ─────────────────────────────────────────────────────────────────────────────
# モデル管理
# ─────────────────────────────────────────────────────────────────────────────

def _list_bilt_models():
    """BILTモデルディレクトリから利用可能なモデルを列挙する。

    Flat layout: bilt_models/*.pth
    Sidecar files: bilt_models/{stem}_params.json, bilt_models/{stem}_rating.json
    """
    models = []
    if not os.path.isdir(BILT_MODELS_DIR):
        return models

    for entry in os.listdir(BILT_MODELS_DIR):
        if not entry.endswith('.pth'):
            continue
        entry_path = os.path.join(BILT_MODELS_DIR, entry)
        if not os.path.isfile(entry_path):
            continue
        stem = entry[:-4]  # strip .pth
        has_params = os.path.isfile(os.path.join(BILT_MODELS_DIR, f'{stem}_params.json'))
        has_rating = os.path.isfile(os.path.join(BILT_MODELS_DIR, f'{stem}_rating.json'))
        models.append({
            'name':       entry,
            'path':       entry_path,
            'size_mb':    round(os.path.getsize(entry_path) / 1048576, 2),
            'modified':   datetime.fromtimestamp(os.path.getmtime(entry_path)).strftime('%Y-%m-%d %H:%M:%S'),
            'has_params': has_params,
            'has_rating': has_rating,
        })

    return sorted(models, key=lambda m: m['modified'], reverse=True)


@app.route('/api/bilt/models')
def list_models():
    return jsonify({'models': _list_bilt_models()})


@app.route('/api/bilt/models/params')
def get_model_params():
    """Return params for a named model (flat layout: {stem}_params.json)."""
    name = request.args.get('name', '').strip()
    if not name:
        return jsonify({'error': 'name parameter required'}), 400
    stem = name[:-4] if name.endswith('.pth') else name
    params_file = os.path.join(BILT_MODELS_DIR, f'{stem}_params.json')
    if not os.path.isfile(params_file):
        return jsonify({'error': 'params not found for this model'}), 404
    try:
        with open(params_file) as fh:
            return jsonify({'params': json.load(fh)})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/bilt/model/load', methods=['POST'])
def load_model():
    global current_bilt_model, current_bilt_model_name, model_info
    data = request.json or {}
    model_name = data.get('model_name', '')
    if not model_name:
        return jsonify({'error': 'model_name が必要です'}), 400

    model_path = os.path.join(BILT_MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        return jsonify({'error': f'モデルが見つかりません: {model_name}'}), 404

    try:
        from bilt import BILT
        bilt_obj = BILT(model_path)
        current_bilt_model = bilt_obj
        current_bilt_model_name = model_name
        model_info = {
            'name': model_name,
            'classes': bilt_obj.class_names or [],
            'loaded': True,
            'variant': bilt_obj.variant,
        }
        logger.info(f'BILT モデルをロードしました: {model_name}')
        return jsonify({'success': True, 'model_info': model_info})
    except Exception as e:
        logger.error(f'BILT モデルのロードに失敗しました: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/api/bilt/model/info')
def model_info_route():
    return jsonify(model_info)


# ─────────────────────────────────────────────────────────────────────────────
# カメラ管理
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/bilt/cameras')
def get_cameras():
    cameras = []
    if sys.platform == 'win32':
        for i in range(4):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    cameras.append({'index': i, 'name': f'Camera {i}'})
                cap.release()
            except Exception:
                pass
    elif sys.platform.startswith('linux'):
        import glob as _glob
        nodes = sorted(_glob.glob('/dev/video*'))
        for node in nodes:
            try:
                idx = int(node.replace('/dev/video', ''))
                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                if cap.isOpened():
                    cameras.append({'index': idx, 'name': f'Camera {idx}'})
                cap.release()
            except Exception:
                pass
    else:
        for i in range(4):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    cameras.append({'index': i, 'name': f'Camera {i}'})
                cap.release()
            except Exception:
                pass
    return jsonify({'cameras': cameras})


@app.route('/api/bilt/camera/select', methods=['POST'])
def select_camera():
    global _bilt_selected_camera_index
    data = request.json or {}
    idx = data.get('camera_index', 0)
    ok = camera_manager.initialize_camera(idx, app.config)
    if ok:
        _bilt_selected_camera_index = idx   # persist across stop/start cycles
        return jsonify({'success': True})
    return jsonify({'error': 'カメラの初期化に失敗しました'}), 400


# ─────────────────────────────────────────────────────────────────────────────
# 物体検知 (BILT)
# ─────────────────────────────────────────────────────────────────────────────

def _bilt_capture_loop():
    """カメラから常に最新フレームを取得するスレッド。

    推論スレッドとは独立して動作し、カメラバッファを継続的に読み取ります。
    これにより:
      - 推論が遅くてもカメラバッファが古いフレームで溢れない
      - MJPEGストリームがカメラのネイティブFPSで更新される
      - 推論スレッドは常に最新フレームで推論できる
    """
    global _capture_frame, latest_frame, latest_frame_size
    while _capture_active:
        frame, _ = camera_manager.get_frame()
        if frame is None:
            time.sleep(0.005)
            continue
        # Make the latest raw frame available to the inference loop
        with _capture_lock:
            _capture_frame = frame
        # Push raw frame to MJPEG stream immediately — bboxes/labels are
        # drawn on the JS canvas, so the stream never needs to wait for inference.
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if ok:
            with frame_lock:
                latest_frame      = buf.tobytes()
                latest_frame_size = (frame.shape[1], frame.shape[0])


def _bilt_detection_loop():
    """BILTモデルを使ったカメラ推論ループ (バックグラウンドスレッド)。

    フレーム取得はキャプチャスレッドに委任します。
    このスレッドは推論と検知結果の更新のみを担当します。
    """
    global detection_active, latest_detections, _capture_active, _capture_thread
    # Target inference rate.  When inference is slower, the loop runs at
    # model speed.  A minimum 10 ms yield is always kept to prevent 100% CPU
    # and thermal throttling (which actually slows down inference on embedded
    # hardware like Raspberry Pi).
    frame_interval = 1.0 / app.config.get('BILT_FRAME_RATE_LIMIT', 15)
    _last_fps_time = time.time()
    _frame_count = 0

    while detection_active:
        t0 = time.time()
        try:
            with _capture_lock:
                frame = _capture_frame
            if frame is None:
                time.sleep(0.01)
                continue

            _frame_count += 1
            now = time.time()
            sec = now - _last_fps_time
            if sec >= 1.0:
                detection_stats['fps'] = round(_frame_count / sec, 1)
                _frame_count = 0
                _last_fps_time = now

            if current_bilt_model is None:
                time.sleep(0.01)
                continue

            # Pre-resize in OpenCV before PIL conversion.
            # cv2.resize is ~10x faster than PIL's T.Resize on large frames,
            # saving 10–15 ms per inference call on CPU.
            orig_h, orig_w = frame.shape[:2]
            inp_size = current_bilt_model.inferencer.input_size
            small = cv2.resize(frame, (inp_size, inp_size),
                               interpolation=cv2.INTER_LINEAR) \
                    if (orig_w != inp_size or orig_h != inp_size) else frame
            pil_img = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

            conf = detection_settings.get('conf', 0.80)
            iou  = detection_settings.get('iou', 0.45)
            # Keep BILT's internal inferencer threshold in sync so the
            # confidence slider actually filters results.
            if hasattr(current_bilt_model, 'inferencer'):
                current_bilt_model.inferencer.confidence_threshold = conf
            detections = current_bilt_model.predict(pil_img, conf=conf, iou=iou)

            # Scale bboxes from inp_size space back to original frame coords
            sx, sy = orig_w / inp_size, orig_h / inp_size
            if sx != 1.0 or sy != 1.0:
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    det['bbox'] = [int(x1*sx), int(y1*sy),
                                   int(x2*sx), int(y2*sy)]

            # Class filter
            class_filter = detection_settings.get('classes')
            if class_filter and isinstance(class_filter, list):
                detections = [d for d in detections
                              if d.get('class_name') in class_filter
                              or d.get('class_id') in class_filter]

            # max_det limit
            max_det = int(detection_settings.get('max_det', 5))
            if len(detections) > max_det:
                detections = sorted(detections, key=lambda d: d['score'],
                                    reverse=True)[:max_det]

            # Counter mode — show per-class count for the current frame (not accumulated)
            if detection_settings.get('counter_mode'):
                frame_counts = {}
                for det in detections:
                    cls_name = det.get('class_name', 'unknown')
                    frame_counts[cls_name] = frame_counts.get(cls_name, 0) + 1
                object_counters.clear()
                object_counters.update(frame_counts)

            # Dataset capture — save frame + YOLO-format labels to project folder
            if detection_settings.get('dataset_capture') and detections:
                with _capture_lock:
                    raw_frame = _capture_frame
                if raw_frame is not None:
                    ImageManager.save_dataset_image(
                        raw_frame,
                        detections,
                        detection_settings.get('project_folder', ''),
                        app.config,
                    )

            # Chain mode — sequential step detection
            if detection_settings.get('chain_mode'):
                detections, _ = ChainDetectionManager.process_chain_detection(
                    detections, detection_settings, chain_state
                )

            with detection_lock:
                latest_detections = list(detections)
                detection_stats['total_detections'] = len(detections)
                if detections:
                    detection_stats['last_detection_time'] = \
                        datetime.now().isoformat()

        except Exception as e:
            logger.error(f'BILT detection loop error: {e}')
            time.sleep(0.1)

        elapsed = time.time() - t0
        # Always yield ≥10 ms so other threads get CPU time and the CPU
        # doesn't overheat (throttling would make inference even slower).
        time.sleep(max(0.01, frame_interval - elapsed))

    # Stop capture thread before releasing the camera
    _capture_active = False
    if _capture_thread and _capture_thread.is_alive():
        _capture_thread.join(timeout=0.5)
    camera_manager.release()
    logger.info('BILT detection stopped — camera released')


@app.route('/api/bilt/detection/settings', methods=['GET', 'POST'])
def bilt_detection_settings():
    global detection_settings
    if request.method == 'POST':
        data = request.json or {}
        detection_settings.update(data)
        return jsonify({'success': True, 'settings': detection_settings})
    return jsonify(detection_settings)


@app.route('/api/bilt/detection/start', methods=['POST'])
def start_bilt_detection():
    global detection_active, detection_thread, \
           _capture_active, _capture_thread, _capture_frame
    if detection_active:
        return jsonify({'success': True, 'message': '既に実行中です'})
    # Wait for the previous detection thread to finish (it joins + releases camera)
    if detection_thread is not None and detection_thread.is_alive():
        detection_thread.join(timeout=2.0)
    if _bilt_selected_camera_index is None:
        return jsonify({'success': False, 'error': 'カメラが選択されていません'}), 400
    camera_manager.initialize_camera(_bilt_selected_camera_index, app.config)
    if camera_manager.cap is None:
        return jsonify({'success': False, 'error': 'カメラの再初期化に失敗しました'}), 400
    _capture_frame  = None
    _capture_active = True
    _capture_thread = threading.Thread(target=_bilt_capture_loop,
                                       daemon=True, name='bilt-capture')
    _capture_thread.start()
    detection_active = True
    detection_thread = threading.Thread(target=_bilt_detection_loop,
                                        daemon=True, name='bilt-detection')
    detection_thread.start()
    return jsonify({'success': True})


@app.route('/api/bilt/detection/stop', methods=['POST'])
def stop_bilt_detection():
    global detection_active, _capture_active
    _capture_active = False   # stop capture thread first
    detection_active = False  # then inference thread
    return jsonify({'success': True})


@app.route('/api/bilt/detection/status')
def bilt_detection_status():
    return jsonify({
        'active': detection_active,
        'model_loaded': model_info['loaded'],
        'model_name': model_info['name'],
    })


@app.route('/api/bilt/detection/stats')
def bilt_detection_stats():
    return jsonify(detection_stats)


# ── カウンター管理 ────────────────────────────────────────────────────────────────

@app.route('/api/bilt/counters')
def bilt_get_counters():
    return jsonify({'success': True, 'counters': object_counters})


@app.route('/api/bilt/counters/reset', methods=['POST'])
def bilt_reset_counters():
    object_counters.clear()
    return jsonify({'success': True})


# ── チェーン検出管理 ──────────────────────────────────────────────────────────────

@app.route('/api/bilt/chain/status')
def bilt_chain_status():
    try:
        return jsonify({'success': True,
                        'status': ChainDetectionManager.get_chain_status(
                            detection_settings, chain_state)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/bilt/chain/control', methods=['POST'])
def bilt_chain_control():
    try:
        action = (request.json or {}).get('action')
        if action == 'start':
            detection_settings['chain_mode'] = True
            ChainDetectionManager.initialize_chain(chain_state)
        elif action == 'stop':
            detection_settings['chain_mode'] = False
            chain_state['active'] = False
        elif action == 'reset':
            ChainDetectionManager.reset_chain(chain_state)
        else:
            return jsonify({'success': False, 'error': f'不明なアクション: {action}'})
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/bilt/chain/config', methods=['GET', 'POST'])
def bilt_chain_config():
    try:
        if request.method == 'POST':
            data = request.json or {}
            for key in ('chain_steps', 'chain_timeout', 'chain_auto_advance', 'chain_pause_time'):
                if key in data:
                    val = data[key]
                    if key in ('chain_timeout', 'chain_pause_time'):
                        val = float(val)
                    detection_settings[key] = val
        cfg = {k: detection_settings[k]
               for k in ('chain_steps', 'chain_timeout', 'chain_auto_advance', 'chain_pause_time')}
        return jsonify({'success': True, 'config': cfg})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/bilt/chain/acknowledge_skip', methods=['POST'])
def bilt_acknowledge_skip():
    try:
        if chain_state.get('skip_pause', False):
            ChainDetectionManager.acknowledge_skip(chain_state)
            return jsonify({'success': True})
        return jsonify({'success': False, 'message': 'スキップは有りません'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/bilt/detections/latest')
def bilt_latest_detections():
    with detection_lock:
        raw = list(latest_detections)
    with frame_lock:
        fw, fh = latest_frame_size
    # Normalise to the same shape as YOLO's /api/detections/latest so that
    # the shared drawLabelOverlay() in the detection page works without changes.
    dets = []
    for d in raw:
        dets.append({
            'bbox':       d.get('bbox'),
            'class_id':   d.get('class_id'),
            'class_name': d.get('class_name'),
            'confidence': d.get('score', d.get('confidence', 0.0)),
        })
    return jsonify({'success': True, 'detections': dets, 'frame_w': fw, 'frame_h': fh})


@app.route('/api/bilt/frame/latest')
def bilt_latest_frame():
    with frame_lock:
        frame_bytes = latest_frame
    if frame_bytes:
        return Response(frame_bytes, mimetype='image/jpeg')
    # ブランクフレームを返す
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank, 'BILT - No Frame', (120, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    ok, buf = cv2.imencode('.jpg', blank)
    return Response(buf.tobytes(), mimetype='image/jpeg')


@app.route('/api/bilt/detection/reset', methods=['POST'])
def bilt_detection_reset():
    """検出を停止しフレーム/検出データを消去してカメラを解放する。ページロード時に呼び出される。"""
    global detection_active, _capture_active, latest_frame, latest_frame_size
    try:
        detection_active = False
        _capture_active = False
        with frame_lock:
            latest_frame = None
            latest_frame_size = (640, 480)
        camera_manager.release()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── デバイス情報 ──────────────────────────────────────────────────────────────

@app.route('/api/bilt/device/info')
def bilt_device_info():
    """CPU・GPU・RAM・推論モデル情報を返す。検出ページのサイドバーで表示。"""
    import platform
    import torch as _torch
    info = {
        'inference_device': 'cpu',
        'torch_version': _torch.__version__,
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': os.cpu_count(),
        'cuda_available': _torch.cuda.is_available(),
        'model_loaded': model_info.get('loaded', False),
        'model_name': model_info.get('name', ''),
        'model_classes': len(model_info.get('classes', [])),
        'gpus': [],
    }
    if _torch.cuda.is_available():
        info['inference_device'] = 'cuda:0'
        for i in range(_torch.cuda.device_count()):
            props = _torch.cuda.get_device_properties(i)
            total_mb = props.total_memory // (1024 * 1024)
            try:
                alloc_mb = _torch.cuda.memory_allocated(i) // (1024 * 1024)
                reserved_mb = _torch.cuda.memory_reserved(i) // (1024 * 1024)
            except Exception:
                alloc_mb = reserved_mb = 0
            info['gpus'].append({
                'index': i,
                'name': props.name,
                'total_mb': total_mb,
                'allocated_mb': alloc_mb,
                'reserved_mb': reserved_mb,
                'compute': f'{props.major}.{props.minor}',
            })
    try:
        import psutil
        vm = psutil.virtual_memory()
        info['ram_total_mb'] = vm.total // (1024 * 1024)
        info['ram_used_mb'] = vm.used // (1024 * 1024)
        info['ram_percent'] = vm.percent
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            info['cpu_freq_mhz'] = round(cpu_freq.current)
        info['cpu_percent'] = psutil.cpu_percent(interval=0.1)
    except ImportError:
        pass
    return jsonify({'success': True, 'info': info})


# ── チェーンファイル管理 ──────────────────────────────────────────────────────

@app.route('/api/bilt/chains/saved')
def bilt_saved_chains():
    try:
        chains_dir = app.config.get('CHAINS_DIR', os.path.join(BASE_DIR, 'chains'))
        chains = []
        if os.path.exists(chains_dir):
            for f in sorted(os.listdir(chains_dir)):
                if not f.endswith('.json'):
                    continue
                try:
                    with open(os.path.join(chains_dir, f)) as fh:
                        d = json.load(fh)
                    chains.append({
                        'name': d.get('name', f[:-5]),
                        'model_name': d.get('model_name', ''),
                        'steps': len(d.get('chain_steps', [])),
                        'created': d.get('created', ''),
                        'filename': f[:-5],
                    })
                except Exception:
                    pass
        return jsonify(chains)  # raw array — template iterates directly
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/bilt/chains/save', methods=['POST'])
def bilt_save_chain():
    try:
        data = request.json or {}
        name = secure_filename((data.get('chain_name') or '').strip())
        if not name:
            return jsonify({'success': False, 'error': 'チェーン名は必須です'})
        chains_dir = app.config.get('CHAINS_DIR', os.path.join(BASE_DIR, 'chains'))
        os.makedirs(chains_dir, exist_ok=True)
        chain_data = {
            'name': name,
            'model_name': data.get('model_name', ''),
            'created': datetime.now().isoformat(),
            'chain_steps': detection_settings.get('chain_steps', []),
            'chain_timeout': detection_settings.get('chain_timeout', 30.0),
            'chain_auto_advance': detection_settings.get('chain_auto_advance', False),
            'chain_pause_time': detection_settings.get('chain_pause_time', 2.0),
        }
        with open(os.path.join(chains_dir, f'{name}.json'), 'w') as f:
            json.dump(chain_data, f, indent=2, ensure_ascii=False)
        return jsonify({'success': True, 'message': f'チェーン「{name}」を保存しました'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/bilt/chains/load', methods=['POST'])
def bilt_load_chain():
    try:
        name = (request.json or {}).get('chain_name', '')
        path = os.path.join(
            app.config.get('CHAINS_DIR', os.path.join(BASE_DIR, 'chains')),
            f'{secure_filename(name)}.json',
        )
        if not os.path.exists(path):
            return jsonify({'success': False, 'error': 'チェーンが見つかりません'})
        with open(path) as f:
            d = json.load(f)
        for key in ('chain_steps', 'chain_timeout', 'chain_auto_advance', 'chain_pause_time'):
            if key in d:
                detection_settings[key] = d[key]
        return jsonify({'success': True, 'chain_data': d,
                        'message': f'チェーン「{name}」をロードしました'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/bilt/chains/delete', methods=['POST'])
def bilt_delete_chain():
    try:
        name = (request.json or {}).get('chain_name', '')
        path = os.path.join(
            app.config.get('CHAINS_DIR', os.path.join(BASE_DIR, 'chains')),
            f'{secure_filename(name)}.json',
        )
        if os.path.exists(path):
            os.remove(path)
            return jsonify({'success': True, 'message': f'チェーン「{name}」を削除しました'})
        return jsonify({'success': False, 'error': 'チェーンが見つかりません'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# トレーニング出力生成ヘルパー
# ─────────────────────────────────────────────────────────────────────────────

def _compute_model_rating(train_result: dict, history: list) -> dict:
    """Compute a 0-100 model quality score with per-criterion breakdown."""
    if not history or not train_result:
        return {'score': 0, 'grade': 'N/A', 'reasons': ['No training data']}

    final_train = train_result.get('final_train_loss', float('inf'))
    final_val   = train_result.get('final_val_loss',   float('inf'))
    best_val    = train_result.get('best_val_loss',    float('inf'))
    n = len(history)

    scores, reasons = [], []

    # 1. Generalisation / overfitting (30 pts)
    if final_train > 0 and final_val > 0:
        ratio = final_val / final_train
        if ratio < 1.1:
            scores.append(30)
            reasons.append(f'Excellent generalisation (val/train={ratio:.2f})')
        elif ratio < 1.5:
            s = max(0, int(30 * (1.5 - ratio) / 0.4))
            scores.append(s)
            reasons.append(f'Mild overfitting (val/train={ratio:.2f})')
        else:
            scores.append(0)
            reasons.append(f'Significant overfitting (val/train={ratio:.2f})')

    # 2. Convergence improvement (30 pts)
    if n >= 3:
        first_val = history[0]['val_loss']
        if first_val > 0:
            improvement = (first_val - best_val) / first_val
            if improvement > 0.5:
                scores.append(30)
                reasons.append(f'Strong convergence ({improvement*100:.0f}% val loss reduction)')
            elif improvement > 0.2:
                scores.append(int(30 * improvement / 0.5))
                reasons.append(f'Moderate convergence ({improvement*100:.0f}% val loss reduction)')
            elif improvement > 0:
                scores.append(5)
                reasons.append(f'Weak convergence ({improvement*100:.0f}% val loss reduction)')
            else:
                scores.append(0)
                reasons.append('Val loss did not improve during training')

    # 3. Training stability (20 pts)
    val_losses = [h['val_loss'] for h in history]
    mean_v = sum(val_losses) / n
    std_v  = (sum((v - mean_v) ** 2 for v in val_losses) / n) ** 0.5
    cv = std_v / mean_v if mean_v > 0 else 1.0
    if cv < 0.05:
        scores.append(20)
        reasons.append(f'Very stable training (CV={cv*100:.1f}%)')
    elif cv < 0.15:
        scores.append(max(0, int(20 * (0.15 - cv) / 0.1)))
        reasons.append(f'Moderately stable training (CV={cv*100:.1f}%)')
    else:
        scores.append(0)
        reasons.append(f'Unstable training (CV={cv*100:.1f}%)')

    # 4. Post-best divergence (20 pts)
    if best_val > 0 and final_val > 0:
        diverge = (final_val - best_val) / best_val
        if diverge < 0.05:
            scores.append(20)
            reasons.append('Model stable after best checkpoint')
        elif diverge < 0.2:
            scores.append(10)
            reasons.append(f'Slight divergence after best ({diverge*100:.0f}%)')
        else:
            scores.append(0)
            reasons.append(f'Significant divergence after best ({diverge*100:.0f}%)')

    total = sum(scores)
    if total >= 80:
        grade = 'Excellent'
    elif total >= 60:
        grade = 'Good'
    elif total >= 40:
        grade = 'Fair'
    else:
        grade = 'Poor'

    return {
        'score':             total,
        'grade':             grade,
        'best_val_loss':     best_val,
        'final_train_loss':  final_train,
        'final_val_loss':    final_val,
        'reasons':           reasons,
    }


def _generate_training_outputs(
    cfg: dict, train_result: dict, out_dir: str, save_path,
    project_path: str, class_names: list, variant: str, device,
):
    """Generate all post-training artefacts inside *out_dir*."""
    import json as _json

    history = training_state.get('epoch_history', [])

    # ── 1. params_used.json ──────────────────────────────────────────────────
    try:
        with open(os.path.join(out_dir, 'params_used.json'), 'w') as fh:
            _json.dump(cfg, fh, indent=2, default=str)
    except Exception as exc:
        logger.warning(f'params_used.json: {exc}')

    # ── 2. training_results.json ─────────────────────────────────────────────
    try:
        results_doc = {
            'variant':          variant,
            'num_classes':      len(class_names),
            'class_names':      class_names,
            'best_val_loss':    train_result.get('best_val_loss'),
            'final_train_loss': train_result.get('final_train_loss'),
            'final_val_loss':   train_result.get('final_val_loss'),
            'training_time_sec': train_result.get('training_time'),
            'num_epochs':       train_result.get('num_epochs'),
            'num_train_images': train_result.get('num_train'),
            'num_val_images':   train_result.get('num_val'),
            'epoch_history':    history,
        }
        with open(os.path.join(out_dir, 'training_results.json'), 'w') as fh:
            _json.dump(results_doc, fh, indent=2)
    except Exception as exc:
        logger.warning(f'training_results.json: {exc}')

    # ── 3. Loss curve PNGs (train + val separately) ──────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        epochs_ax   = [h['epoch'] for h in history]
        train_losses = [h['train_loss'] for h in history]
        val_losses   = [h['val_loss']   for h in history]

        for (data, label, colour, fname) in [
            (train_losses, 'Train Loss', '#3b82f6', 'train_loss.png'),
            (val_losses,   'Val Loss',   '#ef4444', 'val_loss.png'),
        ]:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(epochs_ax, data, color=colour, linewidth=2, label=label)
            ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
            ax.set_title(label); ax.legend(); ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, fname), dpi=100)
            plt.close(fig)
    except Exception as exc:
        logger.warning(f'Loss plots: {exc}')

    # ── 4. anchors_used.json ─────────────────────────────────────────────────
    try:
        from bilt.variants import get_variant_config
        vcfg = get_variant_config(variant)
        anchors_doc = {
            'strides':       [8, 16, 32, 64],
            'anchor_sizes':  vcfg.get('anchor_sizes', []),
            'anchor_scales': list(vcfg.get('anchor_scales', [1.0, 1.26, 1.587])),
            'aspect_ratios': vcfg.get('anchor_aspect_ratios', []),
            'note': (
                'anchor_sizes[i] is the base size (px) for FPN level i at the '
                'model input resolution. Each base is multiplied by every scale '
                'and combined with every aspect ratio to produce all anchors.'
            ),
        }
        with open(os.path.join(out_dir, 'anchors_used.json'), 'w') as fh:
            _json.dump(anchors_doc, fh, indent=2)
    except Exception as exc:
        logger.warning(f'anchors_used.json: {exc}')

    # ── 5. Sample detections + class_stats ───────────────────────────────────
    n_samples = max(1, min(10, int(cfg.get('sample_images', 5))))
    try:
        from bilt.core import DetectionModel
        from bilt.inferencer import Inferencer
        from PIL import ImageDraw

        dm = DetectionModel.load(str(save_path))
        inf = Inferencer(
            model=dm.model,
            class_names=class_names,
            confidence_threshold=float(cfg.get('conf_threshold', 0.15)),
            nms_threshold=float(cfg.get('iou_threshold', 0.45)),
            input_size=dm.model.input_size,
            device=device,
        )

        val_img_dir = os.path.join(project_path, 'val', 'images')
        if not os.path.isdir(val_img_dir):
            val_img_dir = os.path.join(project_path, 'train', 'images')

        img_files = sorted([
            f for f in os.listdir(val_img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]) if os.path.isdir(val_img_dir) else []

        sample_dir = os.path.join(out_dir, 'sample_detections')
        os.makedirs(sample_dir, exist_ok=True)

        class_counts: dict = {}
        COLOURS = ['#ef4444','#3b82f6','#22c55e','#f59e0b','#a855f7',
                   '#14b8a6','#f97316','#ec4899','#6366f1','#84cc16']

        for idx, fname in enumerate(img_files):
            img = Image.open(os.path.join(val_img_dir, fname)).convert('RGB')
            dets = inf.detect(img)

            for d in dets:
                cn = d['class_name']
                class_counts[cn] = class_counts.get(cn, 0) + 1

            if idx < n_samples:
                draw = ImageDraw.Draw(img)
                for d in dets:
                    x1, y1, x2, y2 = d['bbox']
                    col = COLOURS[d['class_id'] % len(COLOURS)]
                    draw.rectangle([x1, y1, x2, y2], outline=col, width=3)
                    lbl = f"{d['class_name']} {d['score']:.2f}"
                    tw, th = 6 * len(lbl), 14
                    draw.rectangle([x1, max(0, y1 - th - 2), x1 + tw, y1], fill=col)
                    draw.text((x1 + 2, max(0, y1 - th)), lbl, fill='white')
                out_name = f'{idx+1:03d}_{fname}'
                img.save(os.path.join(sample_dir, out_name), quality=90)

        with open(os.path.join(out_dir, 'class_stats.json'), 'w') as fh:
            _json.dump({
                'val_detections_per_class': class_counts,
                'total_val_images_scanned': len(img_files),
                'confidence_threshold': float(cfg.get('conf_threshold', 0.15)),
            }, fh, indent=2)
    except Exception as exc:
        logger.warning(f'Sample detections / class_stats: {exc}')

    # ── 6. model_rating.json ─────────────────────────────────────────────────
    try:
        rating = _compute_model_rating(train_result, history)
        with open(os.path.join(out_dir, 'model_rating.json'), 'w') as fh:
            _json.dump(rating, fh, indent=2)
        with training_state_lock:
            training_state['model_rating'] = rating
        logger.info(
            f'Model rating: {rating["score"]}/100 ({rating["grade"]})'
        )
    except Exception as exc:
        logger.warning(f'model_rating.json: {exc}')

    with training_state_lock:
        training_state['log_lines'].append(
            f'出力を {out_dir} に保存しました '
            f'(best.pth, last.pth, グラフ, サンプル, レーティング)'
        )


# ─────────────────────────────────────────────────────────────────────────────
# トレーニング (BILT)
# ─────────────────────────────────────────────────────────────────────────────

def _run_bilt_training(cfg: dict):
    """BILTトレーニングをバックグラウンドスレッドで実行する。"""
    global training_active
    try:
        with training_state_lock:
            training_state['active'] = True
            training_state['phase'] = 'preparing'
            training_state['error'] = None
            training_state['epoch'] = 0
            training_state['epoch_history'] = []
            training_state['log_lines'] = []
            training_state['training_time'] = None
            training_state['best_val_loss'] = None
            training_state['model_rating'] = None
            training_state['num_train'] = None
            training_state['num_val'] = None
            training_state['output_dir'] = None

        project_path = cfg.get('project_path', '')
        if not project_path or not os.path.isdir(project_path):
            raise ValueError(f'プロジェクトパスが無効です: {project_path}')

        # ── 設定値を取得 ──────────────────────────────────────────────────────
        epochs           = int(cfg.get('epochs', 50))
        batch_size       = max(2, int(cfg.get('batch', 4)))
        variant          = cfg.get('variant', 'core')
        lr               = float(cfg.get('lr0', 2e-3))
        img_size         = int(cfg.get('imgsz', 0)) or None   # 0 → variant default
        workers          = int(cfg.get('workers', 0))
        device_str       = cfg.get('device') or None

        # Training loop
        warmup_epochs    = int(cfg.get('warmup_epochs', 3))
        lr_warmup_epochs = int(cfg.get('lr_warmup_epochs', 3))
        backbone_lr_mult = float(cfg.get('backbone_lr_mult', 0.1))
        weight_decay     = float(cfg.get('weight_decay', 1e-4))
        cos_lr_min       = float(cfg.get('cos_lr_min', 1e-6))
        grad_clip        = float(cfg.get('grad_clip', 5.0))

        # Loss
        focal_alpha      = float(cfg.get('focal_alpha', 0.25))
        focal_gamma      = float(cfg.get('focal_gamma', 2.0))
        box_loss_weight  = float(cfg.get('box_loss_weight', 1.0))
        use_ciou         = bool(cfg.get('use_ciou', False))

        # Augmentation
        augment          = bool(cfg.get('augment', True))
        flip_prob        = float(cfg.get('flip_prob', 0.5))
        color_jitter     = (
            float(cfg.get('color_jitter_b', 0.4)),
            float(cfg.get('color_jitter_c', 0.4)),
            float(cfg.get('color_jitter_s', 0.4)),
            float(cfg.get('color_jitter_h', 0.1)),
        )
        cache_images     = bool(cfg.get('cache_images', False))
        mosaic           = bool(cfg.get('mosaic', False))
        mosaic_prob      = float(cfg.get('mosaic_prob', 0.5))

        # EMA
        use_ema          = bool(cfg.get('use_ema', False))
        ema_decay        = float(cfg.get('ema_decay', 0.99))
        # Output
        # model_name and sample_images are consumed directly from cfg by helpers

        # ── 出力フォルダをプロジェクト runs/bilt/ に準備 ────────────────────────
        import re as _re
        _raw_name = cfg.get('model_name', '').strip()
        if _raw_name:
            safe_name = _re.sub(r'[^\w\-]', '_', _raw_name)
        else:
            project_name_auto = os.path.basename(project_path)
            safe_name = f'{project_name_auto}_{variant}'

        # Training artefacts go into {project}/runs/bilt/train_YYYYMMDD_HHMMSS/
        _run_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        runs_bilt_dir = os.path.join(project_path, 'runs', 'bilt')
        out_dir = os.path.join(runs_bilt_dir, f'train_{_run_stamp}')
        os.makedirs(out_dir, exist_ok=True)
        save_path      = Path(os.path.join(out_dir, 'best.pth'))
        last_save_path = Path(os.path.join(out_dir, 'last.pth'))

        # Flat model copy destination in bilt_models/ (avoid collisions)
        _flat_name = safe_name
        if os.path.exists(os.path.join(BILT_MODELS_DIR, f'{_flat_name}.pth')):
            _i = 2
            while os.path.exists(os.path.join(BILT_MODELS_DIR, f'{_flat_name}_{_i}.pth')):
                _i += 1
            _flat_name = f'{_flat_name}_{_i}'

        # ── クラス名を取得 ─────────────────────────────────────────────────────
        classes_file = os.path.join(project_path, 'classes.txt')
        class_names = []
        if os.path.exists(classes_file):
            with open(classes_file) as f:
                class_names = [l.strip() for l in f if l.strip()]
        if not class_names:
            class_names = ['object']
        num_classes = len(class_names)

        with training_state_lock:
            training_state['phase'] = 'loading_data'
            training_state['log_lines'].append(f'データを読み込んでいます: {project_path}')

        import torch
        device = torch.device(device_str) if device_str else \
                 torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ── Trainer を使って学習 ────────────────────────────────────────────────
        # Trainer handles: backbone freeze/unfreeze, LR warmup ramp, EMA,
        # CIoU loss, mosaic augmentation, image caching, and best-checkpoint saving.
        from bilt.trainer import Trainer

        with training_state_lock:
            training_state['phase'] = 'training'
            training_state['total_epochs'] = epochs
            training_state['log_lines'].append(
                f'BILT-{variant} トレーニング開始: {num_classes}クラス, {epochs}エポック, '
                f'バッチサイズ={batch_size}'
                + (', CIoU' if use_ciou else '')
                + (', EMA' if use_ema else '')
                + (', モザイク' if mosaic else '')
                + (', キャッシュ' if cache_images else '')
            )

        def _epoch_callback(metrics: dict):
            """Trainer が各エポック終了時に呼び出すコールバック。"""
            if not training_active:
                return True   # signal trainer to stop early

            epoch_n    = metrics['epoch']
            total      = metrics['total_epochs']
            train_loss = metrics['train_loss']
            val_loss   = metrics['val_loss']
            current_lr = metrics['lr']

            log_line = (
                f'Epoch {epoch_n}/{total}  '
                f'train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={current_lr:.2e}'
            )
            with training_state_lock:
                training_state['epoch']     = epoch_n
                training_state['loss']      = train_loss
                training_state['val_loss']  = val_loss
                training_state['metrics']   = {
                    'train_loss': train_loss,
                    'val_loss':   val_loss,
                    'lr':         current_lr,
                }
                training_state['log_lines'].append(log_line)
                training_state['epoch_history'].append({
                    'epoch':      epoch_n,
                    'train_loss': train_loss,
                    'val_loss':   val_loss,
                    'lr':         current_lr,
                })
                if len(training_state['log_lines']) > _MAX_LOG_LINES:
                    training_state['log_lines'] = \
                        training_state['log_lines'][-_MAX_LOG_LINES:]

        trainer = Trainer(
            dataset_path     = Path(project_path),
            num_classes      = num_classes,
            class_names      = class_names,
            batch_size       = batch_size,
            learning_rate    = lr,
            num_epochs       = epochs,
            num_workers      = workers,
            input_size       = img_size,
            device           = device,
            variant          = variant,
            warmup_epochs    = warmup_epochs,
            lr_warmup_epochs = lr_warmup_epochs,
            backbone_lr_mult = backbone_lr_mult,
            weight_decay     = weight_decay,
            cos_lr_min       = cos_lr_min,
            grad_clip        = grad_clip,
            focal_alpha      = focal_alpha,
            focal_gamma      = focal_gamma,
            box_loss_weight  = box_loss_weight,
            use_ciou         = use_ciou,
            augment          = augment,
            flip_prob        = flip_prob,
            color_jitter     = color_jitter,
            cache_images     = cache_images,
            mosaic           = mosaic,
            mosaic_prob      = mosaic_prob,
            use_ema          = use_ema,
            ema_decay        = ema_decay,
        )

        # Run — the trainer saves best.pt and last.pt internally
        train_result = trainer.train(
            save_path=save_path,
            last_save_path=last_save_path,
            callback=_epoch_callback,
        )

        # If stop was requested, mark as stopped and exit
        if not training_active:
            with training_state_lock:
                training_state['phase'] = 'stopped'
                training_state['active'] = False
            return

        # ── Post-training: generate all output artefacts ──────────────────────
        _generate_training_outputs(
            cfg, train_result, out_dir, save_path,
            project_path, class_names, variant, device,
        )

        # ── best.pth を bilt_models/ にフラットコピー ──────────────────────────
        _flat_pth = os.path.join(BILT_MODELS_DIR, f'{_flat_name}.pth')
        try:
            import shutil as _shutil
            _shutil.copy2(str(save_path), _flat_pth)
            _params_src = os.path.join(out_dir, 'params_used.json')
            if os.path.isfile(_params_src):
                _shutil.copy2(_params_src,
                              os.path.join(BILT_MODELS_DIR, f'{_flat_name}_params.json'))
            _rating_src = os.path.join(out_dir, 'model_rating.json')
            if os.path.isfile(_rating_src):
                _shutil.copy2(_rating_src,
                              os.path.join(BILT_MODELS_DIR, f'{_flat_name}_rating.json'))
            logger.info(f'bilt_models/{_flat_name}.pth にコピーしました')
        except Exception as _copy_exc:
            logger.error(f'bilt_models/ へのコピーに失敗しました: {_copy_exc}')
            _flat_pth = str(save_path)   # fall back to runs path

        with training_state_lock:
            training_state['phase'] = 'completed'
            training_state['active'] = False
            training_state['model_path'] = _flat_pth
            training_state['output_dir'] = out_dir
            training_state['training_time'] = train_result.get('training_time', 0) if train_result else 0
            training_state['best_val_loss'] = train_result.get('best_val_loss', None) if train_result else None
            training_state['num_train'] = train_result.get('num_train') if train_result else None
            training_state['num_val'] = train_result.get('num_val') if train_result else None

    except Exception as e:
        logger.error(f'BILT training error: {e}\n{traceback.format_exc()}')
        with training_state_lock:
            training_state['phase'] = 'error'
            training_state['error'] = str(e)
            training_state['active'] = False
    finally:
        training_active = False


@app.route('/bilt/train/start', methods=['POST'])
def start_bilt_training():
    global training_active, training_thread
    # Auto-reset stale flag: thread finished but flag wasn't cleared
    if training_active and training_thread and not training_thread.is_alive():
        training_active = False
    if training_active or (training_thread and training_thread.is_alive()):
        return jsonify({'error': '既にトレーニング中です'}), 400

    cfg = request.json or {}
    if not cfg.get('project_path'):
        return jsonify({'error': 'project_path が必要です'}), 400

    training_active = True
    training_thread = threading.Thread(target=_run_bilt_training, args=(cfg,), daemon=True, name='bilt-training')
    training_thread.start()
    return jsonify({'success': True, 'message': 'BILTトレーニングを開始しました'})


@app.route('/bilt/train/stop', methods=['POST'])
def stop_bilt_training():
    global training_active
    training_active = False
    with training_state_lock:
        training_state['phase'] = 'stopping'
        training_state['active'] = False   # unblock UI immediately
    return jsonify({'success': True})


@app.route('/bilt/train/status')
def bilt_training_status():
    with training_state_lock:
        state = dict(training_state)
        state['log_lines'] = list(state['log_lines'])
        state['epoch_history'] = list(state['epoch_history'])
    return jsonify(state)


# ─────────────────────────────────────────────────────────────────────────────
# 再ラベリング (BILT) — ユーザーが学習したモデルを使って画像に自動ラベルを付ける
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/bilt/relabel/models', methods=['POST'])
def list_bilt_relabel_models():
    """プロジェクト内および bilt_models/ にある .pth ファイルを一覧表示する。"""
    data = request.json or {}
    project_path = data.get('project_path', '')
    models = []

    # 1. プロジェクト内の runs/bilt/**/best.pth
    if project_path and os.path.isdir(project_path):
        runs_dir = os.path.join(project_path, 'runs', 'bilt')
        if os.path.isdir(runs_dir):
            for root, _, files in os.walk(runs_dir):
                for fn in files:
                    if fn == 'best.pth':
                        fp = os.path.join(root, fn)
                        mtime = os.path.getmtime(fp)
                        models.append({
                            'path': fp,
                            'name': os.path.relpath(fp, project_path),
                            'created_date': datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S'),
                            'timestamp': mtime,
                            'size_mb': round(os.path.getsize(fp) / 1048576, 2),
                            'source': 'project',
                        })

    # 2. bilt_models/*.pth
    if os.path.isdir(BILT_MODELS_DIR):
        for fn in os.listdir(BILT_MODELS_DIR):
            if fn.endswith('.pth'):
                fp = os.path.join(BILT_MODELS_DIR, fn)
                mtime = os.path.getmtime(fp)
                models.append({
                    'path': fp,
                    'name': fn,
                    'created_date': datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S'),
                    'timestamp': mtime,
                    'size_mb': round(os.path.getsize(fp) / 1048576, 2),
                    'source': 'bilt_models',
                })

    models.sort(key=lambda x: x['timestamp'], reverse=True)
    return jsonify({'models': models})


@app.route('/bilt/relabel/start', methods=['POST'])
def start_bilt_relabel():
    """
    学習済み BILT モデルを使って画像を自動ラベリングする。
    YOLO 形式 (class_id x_c y_c w h) のテキストファイルを書き出す。
    """
    cfg = request.json or {}
    model_path = cfg.get('model_path', '')
    project_path = cfg.get('project_path', '')
    target_split = cfg.get('target_split', 'train')
    conf = float(cfg.get('conf_threshold', 0.25))
    iou = float(cfg.get('iou_threshold', 0.45))
    mode = cfg.get('mode', 'all')          # 'all' | 'labeled' | 'unlabeled'
    backup = cfg.get('backup_enabled', True)

    if not model_path or not os.path.exists(model_path):
        return jsonify({'error': f'モデルが見つかりません: {model_path}'}), 400
    if not project_path or not os.path.isdir(project_path):
        return jsonify({'error': f'プロジェクトパスが無効です: {project_path}'}), 400

    images_dir = os.path.join(project_path, f'{target_split}/images')
    labels_dir = os.path.join(project_path, f'{target_split}/labels')
    if not os.path.isdir(images_dir):
        return jsonify({'error': f'{target_split} に画像フォルダがありません'}), 400
    os.makedirs(labels_dir, exist_ok=True)

    try:
        # バックアップ
        if backup and os.path.isdir(labels_dir) and os.listdir(labels_dir):
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            import shutil
            shutil.copytree(labels_dir,
                            os.path.join(project_path, f'bilt_relabel_backup_{ts}', f'{target_split}/labels'))

        # モデルロード
        from bilt import BILT
        bilt_obj = BILT(model_path)
        bilt_obj.inferencer.confidence_threshold = conf
        bilt_obj.inferencer.nms_threshold = iou

        count = 0
        skipped = 0
        for img_file in sorted(os.listdir(images_dir)):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            lbl_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + '.txt')
            has_lbl = os.path.exists(lbl_path)

            if mode == 'labeled' and not has_lbl:
                skipped += 1
                continue
            if mode == 'unlabeled' and has_lbl:
                skipped += 1
                continue

            img_full = os.path.join(images_dir, img_file)
            detections = bilt_obj.predict(img_full, conf=conf, iou=iou)

            # YOLO 形式に変換して書き込む
            from PIL import Image as _PILImage
            with _PILImage.open(img_full) as _img:
                img_w, img_h = _img.size
            with open(lbl_path, 'w') as f:
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    x_c = ((x1 + x2) / 2) / img_w
                    y_c = ((y1 + y2) / 2) / img_h
                    w = (x2 - x1) / img_w
                    h = (y2 - y1) / img_h
                    cid = det['class_id'] - 1   # BILT は1始まり、YOLO形式は0始まり
                    f.write(f'{cid} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n')
            count += 1

        return jsonify({
            'success': True,
            'message': f'再ラベリング完了: {count} 枚処理 (スキップ: {skipped} 枚)',
            'count': count,
            'skipped': skipped,
        })
    except Exception as e:
        logger.error(f'BILT relabel error: {e}\n{traceback.format_exc()}')
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# プロジェクト内 BILT モデルの一覧
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/bilt/project/models', methods=['POST'])
def list_project_bilt_models():
    """プロジェクト内の runs/bilt 以下の best.pth ファイルを一覧表示する。"""
    data = request.json or {}
    project_path = data.get('project_path', '')
    if not project_path or not os.path.isdir(project_path):
        return jsonify({'models': []})
    models = []
    runs_dir = os.path.join(project_path, 'runs', 'bilt')
    if os.path.isdir(runs_dir):
        for root, _, files in os.walk(runs_dir):
            for fn in files:
                if fn == 'best.pth':
                    fp = os.path.join(root, fn)
                    mtime = os.path.getmtime(fp)
                    models.append({
                        'path': fp,
                        'relative_path': os.path.relpath(fp, project_path),
                        'created_date': datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S'),
                        'timestamp': mtime,
                        'size_mb': round(os.path.getsize(fp) / 1048576, 2),
                    })
    models.sort(key=lambda x: x['timestamp'], reverse=True)
    return jsonify({'models': models, 'has_model': bool(models)})


# ─────────────────────────────────────────────────────────────────────────────
# 学習済みモデルのテスト — ユーザーが画像をアップロードして推論結果を確認する
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/bilt/test/image', methods=['POST'])
def test_bilt_image():
    """
    アップロードされた画像に対して BILT モデルで推論を実行する。
    フォームフィールド:
      - image      : 画像ファイル (必須)
      - model_path : 使用する .pth モデルのパス (省略時は学習後のモデルを自動選択)
      - conf       : 信頼度閾値 (デフォルト 0.25)
      - iou        : NMS IoU 閾値 (デフォルト 0.45)
    レスポンス JSON:
      { success, image_b64, detections: [{class_id, class_name, score, bbox}], model_used }
    """
    # ── モデルを決定 ──────────────────────────────────────────────────────────
    model_path = request.form.get('model_path', '').strip()
    if not model_path:
        # 直近のトレーニング結果から自動選択
        with training_state_lock:
            model_path = training_state.get('model_path', '') or ''
    if not model_path or not os.path.exists(model_path):
        # フォールバック: 現在ロード済みのグローバルモデルを使用
        if current_bilt_model is not None:
            bilt_obj = current_bilt_model
            model_used = current_bilt_model_name or 'loaded model'
        else:
            return jsonify({'success': False, 'error': 'テストに使用するモデルがありません。先にトレーニングを実行するか、モデルをロードしてください。'}), 400
    else:
        try:
            from bilt import BILT
            bilt_obj = BILT(model_path)
            model_used = os.path.basename(model_path)
        except Exception as e:
            return jsonify({'success': False, 'error': f'モデルのロードに失敗しました: {e}'}), 500

    # ── 画像を受け取る ────────────────────────────────────────────────────────
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'image フィールドが必要です'}), 400
    f = request.files['image']
    if not f.filename:
        return jsonify({'success': False, 'error': 'ファイルが選択されていません'}), 400

    conf = float(request.form.get('conf', 0.25))
    iou  = float(request.form.get('iou', 0.45))

    try:
        img_bytes = f.read()
        np_arr = np.frombuffer(img_bytes, np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return jsonify({'success': False, 'error': '画像のデコードに失敗しました'}), 400

        pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        detections = bilt_obj.predict(pil_img, conf=conf, iou=iou)

        # ── バウンディングボックスを描画 ──────────────────────────────────────
        orig_w, orig_h = pil_img.size
        inp_size = bilt_obj.inferencer.input_size
        sx, sy = orig_w / inp_size, orig_h / inp_size
        annotated = bgr.copy()
        colors = [
            (86, 180, 233), (230, 159, 0), (0, 158, 115),
            (213, 94, 0),   (0, 114, 178), (204, 121, 167),
            (240, 228, 66),
        ]
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            x1 = int(x1 * sx); y1 = int(y1 * sy)
            x2 = int(x2 * sx); y2 = int(y2 * sy)
            cid = det.get('class_id', 0)
            color = colors[cid % len(colors)]
            label = f"{det.get('class_name', str(cid))} {det['score']:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        import base64 as _b64
        image_b64 = _b64.b64encode(buf.tobytes()).decode('utf-8')

        return jsonify({
            'success': True,
            'image_b64': image_b64,
            'detections': detections,
            'model_used': model_used,
        })
    except Exception as e:
        logger.error(f'BILT test image error: {e}\n{traceback.format_exc()}')
        return jsonify({'success': False, 'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# ワークフローエンジン — BiltPerCameraStream + WorkflowEngine
# ─────────────────────────────────────────────────────────────────────────────

class BiltPerCameraStream:
    """Per-workflow-camera BILT inference thread.

    One instance per unique camera_index in bilt_detection workflow nodes.
    Detections use BILT format: {"bbox":[x1,y1,x2,y2], "score":float, "class_id":int, "class_name":str}
    """

    def __init__(self, camera_index, model_path: str, conf: float = 0.25,
                 iou: float = 0.45, max_det: int = 100):
        self.camera_index = camera_index
        self.model_path   = model_path
        self.conf         = conf
        self.iou          = iou
        self.max_det      = max_det
        self._manager = EnhancedCameraManager()
        self._model   = None
        self._thread  = None
        self._stop    = threading.Event()
        self._lock    = threading.Lock()
        self.detections: list = []
        self.frame    = None
        self.fps: float = 0.0
        self.ready    = threading.Event()

    def start(self) -> bool:
        if not self._manager.initialize_camera(self.camera_index, app.config):
            logger.error(f'BiltPerCameraStream[{self.camera_index}]: camera init failed')
            return False
        full_path = self.model_path
        if not os.path.isabs(full_path):
            candidate = os.path.join(BILT_MODELS_DIR, self.model_path)
            if not candidate.endswith('.pth'):
                candidate += '.pth'
            if os.path.exists(candidate):
                full_path = candidate
        try:
            if BILT_REPO_DIR not in sys.path:
                sys.path.insert(0, BILT_REPO_DIR)
            from bilt import BILT
            self._model = BILT(full_path)
            logger.info(f'BiltPerCameraStream[{self.camera_index}]: loaded {full_path}')
        except Exception as e:
            logger.error(f'BiltPerCameraStream[{self.camera_index}]: model load failed: {e}')
            self._manager.release()
            return False
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True,
            name=f'bilt-cam-stream-{self.camera_index}',
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=4)
        self._manager.release()

    def get_detections(self) -> list:
        with self._lock:
            return list(self.detections)

    def get_frame_jpeg(self, quality: int = 75):
        with self._lock:
            frame = self.frame
        if frame is None:
            return None
        try:
            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return buf.tobytes() if ok else None
        except Exception:
            return None

    def _loop(self) -> None:
        frame_time = 1.0 / app.config.get('BILT_FRAME_RATE_LIMIT', 15)
        while not self._stop.is_set():
            t0 = time.time()
            try:
                frame, fps = self._manager.get_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue
                orig_h, orig_w = frame.shape[:2]
                inp_size = self._model.inferencer.input_size
                small = cv2.resize(frame, (inp_size, inp_size),
                                   interpolation=cv2.INTER_LINEAR) \
                        if (orig_w != inp_size or orig_h != inp_size) else frame
                pil_img = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                dets = self._model.predict(
                    pil_img, conf=self.conf, iou=self.iou, max_det=self.max_det,
                )
                sx, sy = orig_w / inp_size, orig_h / inp_size
                if sx != 1.0 or sy != 1.0:
                    for d in dets:
                        x1, y1, x2, y2 = d['bbox']
                        d['bbox'] = [int(x1 * sx), int(y1 * sy),
                                     int(x2 * sx), int(y2 * sy)]
                annotated = frame.copy()
                for d in dets:
                    x1, y1, x2, y2 = [int(v) for v in d.get('bbox', [0, 0, 0, 0])]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 128, 0), 2)
                    cv2.putText(
                        annotated,
                        f"{d.get('class_name', '')} {d.get('score', 0):.2f}",
                        (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1,
                    )
                with self._lock:
                    self.detections = dets
                    self.frame      = annotated
                    self.fps        = fps
                self.ready.set()
            except Exception as e:
                logger.error(f'BiltPerCameraStream[{self.camera_index}] loop error: {e}')
                time.sleep(0.1)
            time.sleep(max(0.0, frame_time - (time.time() - t0)))


# ── ワークフロー BILT ストリーム管理 ───────────────────────────────────────────

_bilt_wf_streams: dict = {}
_bilt_wf_streams_lock = threading.Lock()


def _wf_start_bilt_streams(graph: dict) -> None:
    """Start one BiltPerCameraStream per unique camera in bilt_detection nodes."""
    _wf_stop_bilt_streams()
    nodes = graph.get('nodes', [])
    seen: dict = {}
    for node in nodes:
        if node.get('type') != 'bilt_detection':
            continue
        cfg = node.get('config', {})
        cam_idx = cfg.get('camera_index')
        model = (cfg.get('model') or '').strip()
        # Fall back to the globally loaded BILT model when none is set on the node
        if not model:
            model = (current_bilt_model_name or '').strip()
        if cam_idx is None or not model:
            logger.warning(f'bilt_detection node skipped: cam={cam_idx} model="{model}" — assign a model in the node config')
            continue
        seen[int(cam_idx)] = model

    with _bilt_wf_streams_lock:
        for cam_idx, model_path in seen.items():
            if cam_idx in _bilt_wf_streams:
                continue
            conf = float(graph.get('bilt_conf', 0.25))
            iou  = float(graph.get('bilt_iou', 0.45))
            stream = BiltPerCameraStream(cam_idx, model_path, conf=conf, iou=iou)
            if stream.start():
                _bilt_wf_streams[cam_idx] = stream
                logger.info(f'BiltPerCameraStream[{cam_idx}] started with model {model_path}')
            else:
                logger.error(f'BiltPerCameraStream[{cam_idx}] failed to start')


def _wf_stop_bilt_streams() -> None:
    with _bilt_wf_streams_lock:
        for stream in _bilt_wf_streams.values():
            try:
                stream.stop()
            except Exception:
                pass
        _bilt_wf_streams.clear()


# ── WorkflowEngine ─────────────────────────────────────────────────────────

WORKFLOWS_DIR = os.path.join(BASE_DIR, 'workflows')


class WorkflowEngine:
    """グラフベースのワークフロー実行エンジン (BILT 検出のみ)。"""

    def __init__(self):
        self._lock = threading.Lock()
        self.workflow = None
        self.status = 'idle'
        self.current_node_id = None
        self.log = []
        self._thread = None
        self._stop_event = threading.Event()
        self._resume_event = threading.Event()
        self._resume_event.set()
        self._camera_active = False

    def load(self, workflow_data):
        with self._lock:
            if self.status == 'running':
                return False, 'Workflow is running – stop it first'
            self.workflow = workflow_data
            return True, 'ok'

    def get_status(self):
        with self._lock:
            return {
                'status': self.status,
                'current_node_id': self.current_node_id,
                'log': list(self.log[-100:]),
                'camera_active': self._camera_active,
            }

    def start(self):
        prev = self._thread
        if prev and prev.is_alive():
            prev.join(timeout=5)
        with self._lock:
            if self.status == 'running':
                return False, 'Already running'
            if not self.workflow:
                return False, 'ワークフローがロードされていません'
            self._stop_event.clear()
            self._resume_event.set()
            self.status = 'running'
            self.log = []
            self.current_node_id = None
            self._camera_active = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True, 'ok'

    def stop(self):
        self._stop_event.set()
        self._resume_event.set()
        with self._lock:
            self.status = 'idle'
            self._camera_active = False
        _wf_stop_bilt_streams()

    def resume(self):
        with self._lock:
            if self.status == 'paused':
                self.status = 'running'
        self._resume_event.set()

    def _log(self, level, msg):
        entry = {
            'level': level,
            'msg': msg,
            'time': datetime.now().strftime('%H:%M:%S'),
        }
        with self._lock:
            self.log.append(entry)
        logger.info(f"[ワークフロー] [{level}] {msg}")

    def _get_node(self, node_id):
        for n in (self.workflow or {}).get('nodes', []):
            if n['id'] == node_id:
                return n
        return None

    def _get_edges_from(self, node_id, port):
        return [e for e in (self.workflow or {}).get('edges', [])
                if e['from_node'] == node_id and e.get('from_port') == port]

    def _run(self):
        try:
            start_node = next(
                (n for n in self.workflow.get('nodes', []) if n['type'] == 'start'),
                None
            )
            if not start_node:
                with self._lock:
                    self.status = 'error'
                self._log('error', 'ワークフローの開始ノードが見つかりません')
                return
            self._exec_node(start_node['id'], {})
            with self._lock:
                if self.status == 'running':
                    self.status = 'completed'
                    self._log('success', 'ワークフローが正常に完了しました')
        except Exception as e:
            with self._lock:
                self.status = 'error'
            self._log('error', f'予期しないエラー: {e}')
            logger.error(f"ワークフロー エラー: {traceback.format_exc()}")

    def _exec_node(self, node_id, loop_iters):
        if self._stop_event.is_set():
            return
        self._resume_event.wait()
        if self._stop_event.is_set():
            return

        node = self._get_node(node_id)
        if not node:
            self._log('error', f'ノード {node_id} が見つかりません')
            return

        with self._lock:
            self.current_node_id = node_id

        ntype = node['type']
        cfg = node.get('config', {})
        label = cfg.get('label') or ntype
        out_port = 'out'

        if ntype == 'start':
            self._log('info', '>> ワークフローが開始されました')

        elif ntype == 'bilt_detection':
            self._log('info', f'[BILT検出] {label}')
            out_port = self._exec_bilt_detection(node)

        elif ntype == 'alert':
            self._log('warning', f'[アラート] {label}')
            self._exec_alert(cfg, frame_snap=None)

        elif ntype == 'wait':
            dur = float(cfg.get('duration', 1.0))
            self._log('info', f'[待機] {label}: {dur}秒待機')
            self._stop_event.wait(dur)

        elif ntype == 'loop':
            count = loop_iters.get(node_id, 0)
            max_iter = int(cfg.get('max_iterations', 3))
            inf = (max_iter == 0)
            if inf or count < max_iter:
                new_iters = dict(loop_iters)
                new_iters[node_id] = count + 1
                iter_label = f'{count + 1}/{"inf" if inf else max_iter}'
                self._log('info', f'[ループ] {label}: 回数 {iter_label}')
                for edge in self._get_edges_from(node_id, 'body'):
                    self._exec_node(edge['to_node'], new_iters)
                if not self._stop_event.is_set():
                    self._exec_node(node_id, new_iters)
                return
            else:
                loop_iters.pop(node_id, None)
                self._log('info', f'[ループ] {label}: ループ完了 ({max_iter} 回)')

        elif ntype == 'end':
            self._log('success', f'[終了] {label}')
            with self._lock:
                self.status = 'completed'
            return

        edges = self._get_edges_from(node_id, out_port)
        if not edges and out_port != 'out':
            edges = self._get_edges_from(node_id, 'out')

        if len(edges) == 1:
            self._exec_node(edges[0]['to_node'], loop_iters)
        elif len(edges) > 1:
            threads = [
                threading.Thread(
                    target=self._exec_node,
                    args=(e['to_node'], dict(loop_iters)),
                    daemon=True,
                )
                for e in edges
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

    def _exec_bilt_detection(self, node):
        cfg = node.get('config', {})
        target_classes = cfg.get('classes', [])
        count_required = int(cfg.get('count_required', 1))
        timeout        = float(cfg.get('timeout', 30.0))
        failure_action = cfg.get('failure_action', 'alert_pause')
        label          = cfg.get('label', 'BILT Detection')
        cam_idx        = cfg.get('camera_index', None)

        stream = None
        if cam_idx is not None:
            with _bilt_wf_streams_lock:
                stream = _bilt_wf_streams.get(int(cam_idx))
            if stream:
                self._log('info', f'  BILT カメラストリーム cam{cam_idx} (専用スレッド)')
                stream.ready.wait(timeout=5.0)
            else:
                self._log('error',
                          f'  cam{cam_idx} のカメラストリームが起動していません。'
                          f'ノードにBILTモデルが設定されているか確認してください。')
                return self._handle_failure(node, failure_action, None)

        cls_str = ', '.join(target_classes) if target_classes else 'any class'
        cam_tag = f'cam{cam_idx}' if cam_idx is not None else 'global'
        self._log('info',
                  f'  {count_required}数の検出を待機中× [{cls_str}] を {cam_tag}, タイムアウト {timeout}秒')

        with self._lock:
            self._camera_active = True

        deadline = time.time() + timeout
        try:
            while not self._stop_event.is_set():
                if time.time() > deadline:
                    self._log('warning', f'  タイムアウト: {label}')
                    snap = self._capture_frame_snapshot(stream)
                    return self._handle_failure(node, failure_action, snap)

                dets = stream.get_detections() if stream is not None else []
                count = sum(
                    1 for d in dets
                    if not target_classes or d.get('class_name') in target_classes
                )
                if count >= count_required:
                    self._log('success', f'  OK {label}: {count} 回検出を {cam_tag} で確認')
                    return 'success'

                self._stop_event.wait(0.2)
        finally:
            with self._lock:
                self._camera_active = False

        return 'failure'

    def _handle_failure(self, node, failure_action, snap=None):
        node_id = node['id']

        if failure_action in ('alert_pause', 'alert_retry'):
            for edge in self._get_edges_from(node_id, 'failure'):
                alert_node = self._get_node(edge['to_node'])
                if alert_node and alert_node['type'] == 'alert':
                    self._exec_alert(dict(alert_node.get('config', {})), frame_snap=snap)

        if failure_action == 'alert_pause':
            with self._lock:
                self.status = 'paused'
            self._log('warning', '停止 – 再開ボタンをクリックして続行')
            self._resume_event.clear()
            self._resume_event.wait()
            if self._stop_event.is_set():
                return 'failure'
            with self._lock:
                self.status = 'running'
            return 'failure'

        elif failure_action == 'alert_retry':
            self._log('info', '再試行中...')
            return self._exec_bilt_detection(node)

        else:
            self._log('warning', '  検出に失敗しました。続行中...')
            return 'failure'

    def _capture_frame_snapshot(self, stream=None):
        try:
            if stream is not None:
                jpeg = stream.get_frame_jpeg(quality=80)
                if jpeg:
                    return base64.b64encode(jpeg).decode('utf-8')
                return None
            with frame_lock:
                frame = latest_frame.copy() if latest_frame is not None else None
            if frame is None:
                return None
            ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                return base64.b64encode(buf).decode('utf-8')
        except Exception:
            pass
        return None

    def _exec_alert(self, cfg, frame_snap=None):
        message = cfg.get('message', 'Workflow alert')
        send_snap = cfg.get('send_snapshot', False)
        payload_dict = {
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'source': 'workflow',
        }
        if send_snap and frame_snap:
            payload_dict['frame_snapshot'] = frame_snap
        payload = json.dumps(payload_dict).encode('utf-8')

        if cfg.get('http_enabled'):
            url = (cfg.get('http_url') or '').strip()
            if url:
                try:
                    req = urllib.request.Request(
                        url, data=payload,
                        headers={'Content-Type': 'application/json'},
                        method='POST',
                    )
                    urllib.request.urlopen(req, timeout=5)
                    self._log('info', f'  HTTP アラート -> {url}')
                except Exception as e:
                    self._log('error', f'  HTTP アラート失敗: {e}')

        if cfg.get('udp_enabled'):
            udp_ip   = (cfg.get('udp_ip') or '255.255.255.255').strip()
            udp_port = int(cfg.get('udp_port') or 9999)
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                sock.sendto(payload, (udp_ip, udp_port))
                sock.close()
                self._log('info', f'  UDP アラート -> {udp_ip}:{udp_port}')
            except Exception as e:
                self._log('error', f'  UDP アラート失敗: {e}')

        if cfg.get('popup_enabled'):
            popup_payload = dict(payload_dict)
            popup_payload['popup'] = True
            if frame_snap and 'frame_snapshot' not in popup_payload:
                popup_payload['frame_snapshot'] = frame_snap
            try:
                req = urllib.request.Request(
                    'http://localhost:5000/api/alert_feed',
                    data=json.dumps(popup_payload).encode('utf-8'),
                    headers={'Content-Type': 'application/json'},
                    method='POST',
                )
                urllib.request.urlopen(req, timeout=3)
                self._log('info', '  ポップアップ アラート -> localhost:5000/api/alert_feed')
            except Exception as e:
                self._log('error', f'  ポップアップ アラート失敗: {e}')


workflow_engine = WorkflowEngine()


# ── ワークフロー API ────────────────────────────────────────────────────────────

@app.route('/api/workflow/status')
def workflow_status():
    try:
        return jsonify({'success': True, 'workflow': workflow_engine.get_status()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/workflow/start', methods=['POST'])
def workflow_start():
    try:
        # Release the shared detection camera so workflow per-camera streams can open it
        camera_manager.release()
        data = request.json or {}
        graph = data.get('graph')
        if graph:
            ok, msg = workflow_engine.load(graph)
            if not ok:
                return jsonify({'success': False, 'error': msg})
            _wf_start_bilt_streams(graph)
        ok, msg = workflow_engine.start()
        return jsonify({'success': ok, 'error': None if ok else msg})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/workflow/stop', methods=['POST'])
def workflow_stop():
    try:
        workflow_engine.stop()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/workflow/resume', methods=['POST'])
def workflow_resume():
    try:
        workflow_engine.resume()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/workflow/load_graph', methods=['POST'])
def workflow_load_graph():
    try:
        graph = (request.json or {}).get('graph')
        if not graph:
            return jsonify({'success': False, 'error': 'グラフが提供されていません'})
        ok, msg = workflow_engine.load(graph)
        return jsonify({'success': ok, 'error': None if ok else msg})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/workflows/saved')
def saved_workflows():
    try:
        os.makedirs(WORKFLOWS_DIR, exist_ok=True)
        workflows = []
        for f in os.listdir(WORKFLOWS_DIR):
            if f.endswith('.json'):
                try:
                    with open(os.path.join(WORKFLOWS_DIR, f)) as fh:
                        d = json.load(fh)
                    workflows.append({
                        'name': d.get('name', f[:-5]),
                        'filename': f[:-5],
                        'created': d.get('created', ''),
                        'node_count': len(d.get('nodes', [])),
                        'description': d.get('description', ''),
                    })
                except Exception:
                    pass
        return jsonify({'success': True, 'workflows': workflows})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/workflows/save', methods=['POST'])
def save_workflow():
    try:
        data = request.json or {}
        name = secure_filename((data.get('name') or '').strip())
        if not name:
            return jsonify({'success': False, 'error': '名前は必須です'})
        os.makedirs(WORKFLOWS_DIR, exist_ok=True)
        wf = data.get('workflow', {})
        wf['name'] = name
        wf['created'] = datetime.now().isoformat()
        with open(os.path.join(WORKFLOWS_DIR, f'{name}.json'), 'w') as f:
            json.dump(wf, f, indent=2)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/workflows/load_file', methods=['POST'])
def load_workflow_file():
    try:
        name = secure_filename((request.json or {}).get('name', ''))
        path = os.path.join(WORKFLOWS_DIR, f'{name}.json')
        if not os.path.exists(path):
            return jsonify({'success': False, 'error': 'ワークフローが見つかりません'})
        with open(path) as f:
            wf = json.load(f)
        return jsonify({'success': True, 'workflow': wf})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/workflows/delete', methods=['POST'])
def delete_workflow():
    try:
        name = secure_filename((request.json or {}).get('name', ''))
        path = os.path.join(WORKFLOWS_DIR, f'{name}.json')
        if os.path.exists(path):
            os.remove(path)
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'ワークフローが見つかりません'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/workflow/frame')
def workflow_frame():
    """ワークフローモニター用の最新カメラフレーム。"""
    try:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else np.zeros((240, 320, 3), np.uint8)
        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ret:
            return Response(buf.tobytes(), mimetype='image/jpeg')
        return jsonify({'success': False, 'error': 'エンコードに失敗しました'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/workflow/camera_frame/<int:cam_idx>')
def workflow_camera_frame(cam_idx):
    """ワークフロー内の指定されたカメラからJPEGフレームを返します。"""
    with _bilt_wf_streams_lock:
        stream = _bilt_wf_streams.get(cam_idx)
    if stream is None:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            return '', 204
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            return '', 204
        return Response(buf.tobytes(), mimetype='image/jpeg')
    jpeg = stream.get_frame_jpeg(quality=75)
    if jpeg is None:
        return '', 204
    return Response(jpeg, mimetype='image/jpeg')


@app.route('/api/workflow/streams')
def workflow_streams():
    """アクティブなワークフロー BILT ストリームを一覧表示します。"""
    with _bilt_wf_streams_lock:
        return jsonify({
            'streams': [
                {
                    'camera_index': idx,
                    'model': s.model_path,
                    'fps': round(s.fps, 1),
                    'active': not s._stop.is_set(),
                    'det_count': len(s.detections),
                }
                for idx, s in _bilt_wf_streams.items()
            ]
        })


@app.route('/api/workflow/detections')
def workflow_detections():
    """ワークフローモニターパネルにおける現在のリアルタイム検出状況。"""
    try:
        with _bilt_wf_streams_lock:
            streams_snapshot = dict(_bilt_wf_streams)
        if streams_snapshot:
            dets = []
            for stream in streams_snapshot.values():
                dets.extend(stream.get_detections())
        else:
            with detection_lock:
                dets = list(latest_detections)
        return jsonify({'success': True, 'detections': dets})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# 起動
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = app.config.get('BILT_SERVICE_PORT', 5002)
    logger.info(f'BILT サービスを起動中: port {port}')
    app.run(host='127.0.0.1', port=port, debug=False, threaded=True)
