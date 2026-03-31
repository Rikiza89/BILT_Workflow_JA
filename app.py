# Copyright (C) 2026 Rikiza89
# Licensed under the Apache License, Version 2.0
"""
BILT_Workflow_JA ラベリング + 物体検知一体型アプリケーション　（port 5000を使用）

urlタブ:
  /              → インデックスページ (ラベリングまたは物体検知から選択)
  /annotation/   → ラベリングワークスペース (プロジェクト選択, ラベリング作業, モデルトレーニング)
  /workflow/     → 物体検知のワークフロー設定をカンヴァ上で簡単に出来るツール (製造業向け品質管理など)

bilt_service.pyがポート5002で動作している必要があります。

起動: python app.py
"""

import os
import sys
import json
import cv2
import base64
import threading
import shutil
from datetime import datetime
from pathlib import Path

import eventlet
eventlet.monkey_patch()
from flask import (Flask, render_template, request, jsonify,
                   send_from_directory, redirect, url_for, Response)
from flask_socketio import SocketIO, emit

# ── パス設定 ────────────────────────────────────────────────────────────────
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(os.path.dirname(sys.executable))
    template_folder = os.path.join(sys._MEIPASS, 'templates')
    static_folder = os.path.join(sys._MEIPASS, 'static')
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    template_folder = os.path.join(BASE_DIR, 'templates')
    static_folder = os.path.join(BASE_DIR, 'static')

os.chdir(BASE_DIR)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from bilt_client import BILTClient, check_bilt_service, _conn_err_msg as _bilt_conn_err

# ── Flask / SocketIO 設定 ────────────────────────────────────────────────────
app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
app.config['SECRET_KEY'] = 'your-secret-key'

socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins='*')

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ── BILT サービスクライエント ────────────────────────────────────────────────────
bilt_client = BILTClient('http://127.0.0.1:5002')

# ── ラベリング状況（初期設定）──────────────────────────────────────────────────────────
projects_dir = os.path.join(BASE_DIR, 'projects')
current_project = None


# ─────────────────────────────────────────────────────────────────────────────
# ProjectManager  (ラベリング用ヘルパークラス)
# ─────────────────────────────────────────────────────────────────────────────

class ProjectManager:
    def __init__(self, project_name, project_path):
        self.name = project_name
        self.path = project_path
        self.classes_file = os.path.join(project_path, 'classes.txt')
        self.data_yaml = os.path.join(project_path, 'data.yaml')

    def get_task_type(self):
        cfg_file = os.path.join(self.path, 'project_config.json')
        if os.path.exists(cfg_file):
            with open(cfg_file) as f:
                return json.load(f).get('task_type', 'detect')
        return 'detect'

    def set_task_type(self, task_type):
        if task_type not in ('detect', 'segment', 'obb'):
            return False
        cfg_file = os.path.join(self.path, 'project_config.json')
        cfg = {}
        if os.path.exists(cfg_file):
            with open(cfg_file) as f:
                cfg = json.load(f)
        cfg['task_type'] = task_type
        with open(cfg_file, 'w') as f:
            json.dump(cfg, f, indent=2)
        return True

    def create_structure(self):
        for folder in ('train/images', 'train/labels', 'val/images', 'val/labels'):
            os.makedirs(os.path.join(self.path, folder), exist_ok=True)
        if not os.path.exists(self.classes_file):
            with open(self.classes_file, 'w') as f:
                f.write('object\n')
        cfg_file = os.path.join(self.path, 'project_config.json')
        if not os.path.exists(cfg_file):
            with open(cfg_file, 'w') as f:
                json.dump({'task_type': 'detect'}, f, indent=2)
        self.update_data_yaml()

    def update_data_yaml(self):
        import yaml
        classes = self.get_classes()
        data = {
            'train': os.path.abspath(os.path.join(self.path, 'train/images')),
            'val': os.path.abspath(os.path.join(self.path, 'val/images')),
            'nc': len(classes),
            'names': classes,
        }
        with open(self.data_yaml, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    def get_classes(self):
        if os.path.exists(self.classes_file):
            with open(self.classes_file) as f:
                return [l.strip() for l in f if l.strip()]
        return []

    def save_classes(self, classes):
        with open(self.classes_file, 'w') as f:
            f.writelines(f'{c}\n' for c in classes)
        self.update_data_yaml()

    def get_images(self, split='train'):
        images_dir = os.path.join(self.path, f'{split}/images')
        if not os.path.exists(images_dir):
            return []
        valid = ('.jpg', '.jpeg', '.png', '.bmp')
        images = []
        for fn in os.listdir(images_dir):
            if fn.lower().endswith(valid):
                images.append({
                    'filename': fn,
                    'path': os.path.join(images_dir, fn),
                    'has_labels': self.has_labels(fn, split),
                })
        return sorted(images, key=lambda x: x['filename'])

    def has_labels(self, image_filename, split='train'):
        lbl = os.path.splitext(image_filename)[0] + '.txt'
        return os.path.exists(os.path.join(self.path, f'{split}/labels', lbl))

    def get_labels(self, image_filename, split='train'):
        lbl = os.path.splitext(image_filename)[0] + '.txt'
        lbl_path = os.path.join(self.path, f'{split}/labels', lbl)
        if not os.path.exists(lbl_path):
            return []
        task_type = self.get_task_type()
        labels = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                class_id = int(parts[0])
                if task_type == 'detect' and len(parts) >= 5:
                    labels.append({'class_id': class_id,
                                   'x_center': float(parts[1]), 'y_center': float(parts[2]),
                                   'width': float(parts[3]), 'height': float(parts[4]),
                                   'type': 'detect'})
                elif task_type == 'segment' and len(parts) >= 7:
                    points = [{'x': float(parts[i]), 'y': float(parts[i+1])}
                               for i in range(1, len(parts)-1, 2)]
                    if len(points) >= 3:
                        labels.append({'class_id': class_id, 'points': points, 'type': 'segment'})
                elif task_type == 'obb' and len(parts) == 9:
                    points = [{'x': float(parts[i]), 'y': float(parts[i+1])}
                               for i in range(1, 9, 2)]
                    labels.append({'class_id': class_id, 'points': points, 'type': 'obb'})
        return labels

    def save_labels(self, image_filename, labels, split='train'):
        lbl = os.path.splitext(image_filename)[0] + '.txt'
        lbl_path = os.path.join(self.path, f'{split}/labels', lbl)
        task_type = self.get_task_type()
        with open(lbl_path, 'w') as f:
            for label in labels:
                cid = label['class_id']
                if task_type == 'detect':
                    f.write(f"{cid} {label['x_center']} {label['y_center']} "
                            f"{label['width']} {label['height']}\n")
                elif task_type == 'segment' and len(label.get('points', [])) >= 3:
                    pts = ' '.join(f"{p['x']} {p['y']}" for p in label['points'])
                    f.write(f"{cid} {pts}\n")
                elif task_type == 'obb' and len(label.get('points', [])) == 4:
                    pts = ' '.join(f"{p['x']} {p['y']}" for p in label['points'])
                    f.write(f"{cid} {pts}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CameraManager  (ラベリング用カメラの起動 – 物体検知用カメラとは違うクラスになします)
# ─────────────────────────────────────────────────────────────────────────────

class CameraManager:
    def __init__(self):
        self.camera = None
        self.active = False
        self.color_mode = 'bgr'

    def _detect_color_issue(self, frame):
        if frame is None or len(frame.shape) != 3:
            return None
        b, g, r = frame[:, :, 0].mean(), frame[:, :, 1].mean(), frame[:, :, 2].mean()
        if b > r * 1.5 and b > 100:
            return 'yuv2bgr'
        if r > b * 1.5 and r > 100:
            return 'rgb2bgr'
        return None

    def start_camera(self, camera_id=0, resolution='1080p'):
        try:
            res_map = {'4k': (3840, 2160), '1080p': (1920, 1080),
                       '720p': (1280, 720), '480p': (640, 480), '360p': (480, 360)}
            w, h = res_map.get(resolution, (1920, 1080))

            # ── Raspberry Pi のpicamera2 を使用したカメラモジュール ──────────────────────
            if camera_id == 'picamera2':
                from bilt_managers import _PiCamera2Backend  # カメラタイプ: 無視
                self.camera = _PiCamera2Backend(w, h, 30)
                self.color_mode = 'bgr'
                self.active = True
                return True

            # ── USB / built-in カメラをOpenCV経由で起動（WindowsとLinux用） ──────────────────────────────
            if sys.platform == 'win32':
                self.camera = cv2.VideoCapture(camera_id)
                if not self.camera.isOpened():
                    self.camera = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            elif sys.platform.startswith('linux'):
                self.camera = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
                if not self.camera.isOpened():
                    self.camera = cv2.VideoCapture(camera_id)
            else:
                self.camera = cv2.VideoCapture(camera_id)

            if not self.camera.isOpened():
                return False

            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            if resolution in ('4k', '1080p'):
                self.camera.set(cv2.CAP_PROP_FOURCC,
                                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            aw = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            ah = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if not (aw >= 1920 or ah >= 1080):
                ret, test = self.camera.read()
                if ret:
                    issue = self._detect_color_issue(test)
                    self.color_mode = issue if issue else 'bgr'
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.active = True
            return True
        except Exception as e:
            logger.error(f"app.py - CameraManager.start_camera: {e}")
            return False

    def stop_camera(self):
        if self.camera:
            self.camera.release()
        self.active = False

    def get_available_cameras(self):
        cameras = []

        if sys.platform == 'win32':
            for i in range(4):
                try:
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                    if cap.isOpened():
                        cameras.append(i)
                    cap.release()
                except Exception:
                    pass

        elif sys.platform.startswith('linux'):
            import glob as _glob
            nodes = sorted(_glob.glob('/dev/video*'))
            indices = []
            for node in nodes:
                try:
                    indices.append(int(node.replace('/dev/video', '')))
                except ValueError:
                    pass
            if not indices:
                indices = list(range(4))
            for i in indices:
                try:
                    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
                    if cap.isOpened():
                        cameras.append(i)
                    cap.release()
                except Exception:
                    pass
            # Add picamera2 if available
            try:
                import picamera2  # noqa: F401
                cameras.append('picamera2')
            except ImportError:
                pass

        else:
            for i in range(4):
                try:
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        cameras.append(i)
                    cap.release()
                except Exception:
                    pass

        return cameras

    def set_resolution(self, width, height):
        if self.camera and self.active:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def capture_frame(self):
        if self.camera and self.active:
            ret, frame = self.camera.read()
            if ret and frame is not None:
                if self.color_mode == 'yuv2bgr':
                    frame = frame[:, :, ::-1]
                elif self.color_mode == 'rgb2bgr':
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if app.config.get('CAMERA_FLIP_HORIZONTAL', False):
                    frame = cv2.flip(frame, 1)
                return frame
        return None


annotation_camera = CameraManager()


# ─────────────────────────────────────────────────────────────────────────────
# ルート – ランディングページ
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detection')
def detection_index():
    return render_template('detection_page.html')


# ─────────────────────────────────────────────────────────────────────────────
# ルート – ラベリング
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/annotation/')
@app.route('/annotation')
def annotation_index():
    return redirect(url_for('projects'))


@app.route('/projects')
def projects():
    os.makedirs(projects_dir, exist_ok=True)
    project_list = []
    for name in os.listdir(projects_dir):
        path = os.path.join(projects_dir, name)
        if os.path.isdir(path):
            pm = ProjectManager(name, path)
            project_list.append({
                'name': name,
                'train_images': len(pm.get_images('train')),
                'val_images': len(pm.get_images('val')),
                'classes': len(pm.get_classes()),
            })
    return render_template('projects.html', projects=project_list)


@app.route('/create_project', methods=['POST'])
def create_project():
    name = request.json.get('name')
    if not name:
        return jsonify({'エラー': 'プロジェクト名が必要です'}), 400
    path = os.path.join(projects_dir, name)
    if os.path.exists(path):
        return jsonify({'エラー': 'プロジェクトは既に存在します'}), 400
    pm = ProjectManager(name, path)
    pm.create_structure()
    return jsonify({'success': True, 'message': 'プロジェクトが作成されました'})


@app.route('/load_project/<project_name>')
def load_project(project_name):
    global current_project
    path = os.path.join(projects_dir, project_name)
    if not os.path.exists(path):
        return redirect(url_for('projects'))
    current_project = ProjectManager(project_name, path)
    return redirect(url_for('workspace'))


@app.route('/workspace')
def workspace():
    if not current_project:
        return redirect(url_for('projects'))
    return render_template('annotation_page.html',
                           project=current_project, current_project=current_project)


# カメラ (ラベリング用 – 物体検知用カメラとは違うクラスになしています)
@app.route('/api/camera/available')
def get_available_cameras():
    return jsonify({'cameras': annotation_camera.get_available_cameras()})


@app.route('/api/camera/start', methods=['POST'])
def start_annotation_camera():
    cid = request.json.get('camera_id', 0)
    res = request.json.get('resolution', '1080p')
    if annotation_camera.start_camera(cid, res):
        return jsonify({'success': True})
    return jsonify({'error': 'カメラの起動に失敗しました'}), 400


@app.route('/api/camera/stop', methods=['POST'])
def stop_annotation_camera():
    annotation_camera.stop_camera()
    return jsonify({'success': True})


@app.route('/api/camera/capture', methods=['POST'])
def capture_annotation_image():
    if not current_project:
        return jsonify({'error': 'プロジェクトがロードされていません'}), 400
    frame = annotation_camera.capture_frame()
    if frame is None:
        return jsonify({'error': '画像の取得に失敗しました'}), 400
    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    filename = f'img_{ts}.jpg'
    split = request.json.get('split', 'train')
    img_path = os.path.join(current_project.path, f'{split}/images', filename)
    cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 99])
    return jsonify({'success': True, 'filename': filename,
                    'message': f'画像が {split} に保存されました'})


# Project data
@app.route('/api/project/images/<split>')
def get_project_images(split):
    if not current_project:
        return jsonify({'error': 'プロジェクトがロードされていません'}), 400
    return jsonify({'images': current_project.get_images(split)})


@app.route('/api/project/classes', methods=['GET', 'POST'])
def project_classes():
    if not current_project:
        return jsonify({'error': 'プロジェクトがロードされていません'}), 400
    if request.method == 'POST':
        current_project.save_classes(request.json.get('classes', []))
        return jsonify({'success': True})
    return jsonify({'classes': current_project.get_classes()})


@app.route('/api/labels/<split>/<filename>', methods=['GET', 'POST'])
def labels(split, filename):
    if not current_project:
        return jsonify({'error': 'プロジェクトがロードされていません'}), 400
    if request.method == 'POST':
        current_project.save_labels(filename, request.json.get('labels', []), split)
        return jsonify({'success': True})
    return jsonify({'labels': current_project.get_labels(filename, split)})


@app.route('/api/training/config', methods=['GET', 'POST'])
def training_config():
    if not current_project:
        return jsonify({'error': 'プロジェクトがロードされていません'}), 400
    cfg_file = os.path.join(current_project.path, 'training_config.json')
    if request.method == 'POST':
        with open(cfg_file, 'w') as f:
            json.dump(request.json, f, indent=2)
        return jsonify({'success': True})
    if os.path.exists(cfg_file):
        with open(cfg_file) as f:
            return jsonify(json.load(f))
    return jsonify({})


@app.route('/images/<split>/<filename>')
def serve_image(split, filename):
    if not current_project:
        return 'No project loaded', 404
    return send_from_directory(os.path.join(current_project.path, f'{split}/images'), filename)


@app.route('/api/project/task_type', methods=['GET', 'POST'])
def project_task_type():
    if not current_project:
        return jsonify({'error': 'プロジェクトがロードされていません'}), 400
    if request.method == 'POST':
        tt = request.json.get('task_type')
        if current_project.set_task_type(tt):
            return jsonify({'success': True, 'task_type': tt})
        return jsonify({'error': '無効なタスクタイプ'}), 400
    return jsonify({'task_type': current_project.get_task_type()})


# SocketIO – ラベリング用ライブカメラフィード
@socketio.on('start_camera_feed')
def handle_camera_feed(data):
    cid = data.get('camera_id', 0)
    res = data.get('resolution', '1080p')
    if not annotation_camera.start_camera(cid, res):
        emit('camera_error', {'error': 'カメラの開始に失敗しました'})
        return

    def stream():
        while annotation_camera.active:
            try:
                frame = annotation_camera.capture_frame()
                if frame is not None:
                    h, w = frame.shape[:2]
                    max_dim = 800
                    if w > max_dim or h > max_dim:
                        scale = max_dim / max(w, h)
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    socketio.emit('camera_frame', {
                        'frame': base64.b64encode(buf).decode(),
                        'width': frame.shape[1], 'height': frame.shape[0],
                    })
                socketio.sleep(0.033)
            except Exception as e:
                logger.error(f"カメラストリームエラー: {e}")
                break

    socketio.start_background_task(stream)
    emit('camera_started', {'message': 'ラベリング用カメラフィードが開始されました'})


@socketio.on('stop_camera_feed')
def handle_stop_camera():
    annotation_camera.stop_camera()
    emit('camera_stopped', {'message': 'ラベリング用カメラフィードが停止されました'})


# ── プロジェクト一覧・作成 (検出ページのデータセット保存先に使用) ──────────────────

@app.route('/api/projects')
def api_projects():
    try:
        pdir = projects_dir
        result = []
        if os.path.exists(pdir):
            for item in os.listdir(pdir):
                if os.path.isdir(os.path.join(pdir, item)):
                    result.append(item)
        return jsonify(sorted(result))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/create_project', methods=['POST'])
def api_create_project():
    try:
        from werkzeug.utils import secure_filename as _sf
        data = request.json or {}
        name = _sf((data.get('project_name') or '').strip())
        if not name:
            return jsonify({'success': False, 'error': 'プロジェクト名は必須です'})
        path = os.path.join(projects_dir, name)
        if os.path.exists(path):
            return jsonify({'success': False, 'error': 'プロジェクトは既に存在します'})
        # Minimal annotation-compatible structure
        for sub in ('train/images', 'train/labels', 'val/images', 'val/labels'):
            os.makedirs(os.path.join(path, sub), exist_ok=True)
        import json as _json
        with open(os.path.join(path, 'classes.txt'), 'w') as f:
            f.write('\n'.join(data.get('classes', [])))
        return jsonify({'success': True, 'project': name})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── デバイス情報プロキシ ────────────────────────────────────────────────────────

@app.route('/api/device/info')
def device_info_proxy():
    try:
        return jsonify(bilt_client._get('/api/bilt/device/info'))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── 検出リセットプロキシ ────────────────────────────────────────────────────────

@app.route('/api/bilt/detection/reset', methods=['POST'])
def bilt_detection_reset_proxy():
    try:
        return jsonify(bilt_client._post('/api/bilt/detection/reset'))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── チェーンファイルプロキシ ────────────────────────────────────────────────────

@app.route('/api/bilt/chains/saved')
def bilt_chains_saved_proxy():
    try:
        return jsonify(bilt_client._get('/api/bilt/chains/saved'))
    except Exception as e:
        return jsonify([])


@app.route('/api/bilt/chains/save', methods=['POST'])
def bilt_chains_save_proxy():
    try:
        return jsonify(bilt_client._post('/api/bilt/chains/save', json=request.json))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/bilt/chains/load', methods=['POST'])
def bilt_chains_load_proxy():
    try:
        return jsonify(bilt_client._post('/api/bilt/chains/load', json=request.json))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/bilt/chains/delete', methods=['POST'])
def bilt_chains_delete_proxy():
    try:
        return jsonify(bilt_client._post('/api/bilt/chains/delete', json=request.json))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# BILT カウンター
@app.route('/api/bilt/counters')
def bilt_get_counters():
    try:
        result = bilt_client.get_counters()
        return jsonify(result.get('counters', {}) if result.get('success') else {})
    except Exception:
        return jsonify({})


@app.route('/api/bilt/counters/reset', methods=['POST'])
def bilt_reset_counters():
    try:
        return jsonify(bilt_client.reset_counters())
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# BILT チェーン物体検知
@app.route('/api/bilt/chain/status')
def bilt_chain_status():
    try:
        result = bilt_client.get_chain_status()
        return jsonify(result.get('status', {'active': False}) if result.get('success')
                       else {'active': False, 'error': result.get('error')})
    except Exception as e:
        return jsonify({'active': False, 'error': str(e)})


@app.route('/api/bilt/chain/control', methods=['POST'])
def bilt_chain_control():
    try:
        return jsonify(bilt_client.chain_control((request.json or {}).get('action')))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/bilt/chain/config', methods=['GET', 'POST'])
def bilt_chain_config():
    try:
        if request.method == 'POST':
            return jsonify(bilt_client.update_chain_config(request.json))
        result = bilt_client.get_chain_config()
        return jsonify(result.get('config', {}) if result.get('success') else {})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/bilt/chain/acknowledge_skip', methods=['POST'])
def bilt_acknowledge_skip():
    try:
        return jsonify(bilt_client.acknowledge_skip())
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/detection/workflow')
def detection_workflow():
    return render_template('workflow.html')


# ── ワークフロー API プロクシー ─────────────────────────────────────────────────────────

@app.route('/api/workflow/status')
def wf_status():
    try:
        return jsonify(bilt_client.get_workflow_status())
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/workflow/start', methods=['POST'])
def wf_start():
    try:
        data = request.json or {}
        return jsonify(bilt_client.start_workflow(data.get('graph')))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/workflow/stop', methods=['POST'])
def wf_stop():
    try:
        return jsonify(bilt_client.stop_workflow())
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/workflow/resume', methods=['POST'])
def wf_resume():
    try:
        return jsonify(bilt_client.resume_workflow())
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/workflow/load_graph', methods=['POST'])
def wf_load_graph():
    try:
        graph = (request.json or {}).get('graph')
        return jsonify(bilt_client.load_workflow_graph(graph))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/workflows/saved')
def wf_saved():
    try:
        return jsonify(bilt_client.get_saved_workflows())
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/workflows/save', methods=['POST'])
def wf_save():
    try:
        data = request.json or {}
        return jsonify(bilt_client.save_workflow(data.get('name', ''), data.get('workflow', {})))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/workflows/load_file', methods=['POST'])
def wf_load_file():
    try:
        return jsonify(bilt_client.load_workflow_file((request.json or {}).get('name', '')))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/workflows/delete', methods=['POST'])
def wf_delete():
    try:
        return jsonify(bilt_client.delete_workflow((request.json or {}).get('name', '')))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/workflow/detections')
def wf_detections():
    try:
        return jsonify(bilt_client.get_workflow_detections())
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/workflow/streams')
def wf_streams():
    try:
        return jsonify(bilt_client.get_workflow_streams())
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/workflow/frame')
def wf_frame():
    """ワークフローモニター用にライブフレームをプロクシーします."""
    try:
        import requests as _req
        resp = _req.get('http://127.0.0.1:5002/api/workflow/frame', timeout=3, stream=True)
        return Response(resp.content, mimetype='image/jpeg')
    except Exception:
        import numpy as np
        blank = cv2.imencode('.jpg',
            np.zeros((240, 320, 3), dtype=np.uint8))[1].tobytes()
        return Response(blank, mimetype='image/jpeg')


@app.route('/api/workflow/camera_frame/<int:cam_idx>')
def wf_camera_frame(cam_idx):
    """ワークフローモニター用に特定カメラフレームをプロクシーします."""
    try:
        import requests as _req
        resp = _req.get(f'http://127.0.0.1:5002/api/workflow/camera_frame/{cam_idx}',
                        timeout=3, stream=True)
        return Response(resp.content, mimetype='image/jpeg')
    except Exception:
        import numpy as np
        blank = cv2.imencode('.jpg',
            np.zeros((240, 320, 3), dtype=np.uint8))[1].tobytes()
        return Response(blank, mimetype='image/jpeg')


# ─────────────────────────────────────────────────────────────────────────────
# ヘルプページ
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/help')
def help_page():
    return render_template('help.html')


# ─────────────────────────────────────────────────────────────────────────────
# BILT サービスプロクシー
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/bilt/service/health')
def bilt_service_health():
    return jsonify({'success': True, 'service_available': check_bilt_service()})


# BILT モデル管理
@app.route('/api/bilt/models')
def bilt_models():
    try:
        return jsonify(bilt_client.get_models())
    except Exception as e:
        return jsonify({'models': [], 'error': _bilt_conn_err(e)})


@app.route('/api/bilt/models/params')
def bilt_model_params():
    name = request.args.get('name', '')
    try:
        return jsonify(bilt_client.get_model_params(name))
    except Exception as e:
        return jsonify({'error': _bilt_conn_err(e)}), 500


@app.route('/api/bilt/model/load', methods=['POST'])
def bilt_load_model():
    try:
        name = request.json.get('model_name')
        return jsonify(bilt_client.load_model(name))
    except Exception as e:
        return jsonify({'success': False, 'error': _bilt_conn_err(e)})


@app.route('/api/bilt/model/info')
def bilt_model_info():
    try:
        return jsonify(bilt_client.get_model_info())
    except Exception as e:
        return jsonify({'success': False, 'error': _bilt_conn_err(e)})


# BILT カメラ
@app.route('/api/bilt/cameras')
def bilt_cameras():
    try:
        return jsonify(bilt_client.get_cameras())
    except Exception as e:
        return jsonify({'cameras': [], 'error': _bilt_conn_err(e)})


@app.route('/api/bilt/camera/select', methods=['POST'])
def bilt_select_camera():
    try:
        return jsonify(bilt_client.select_camera(request.json.get('camera_index', 0)))
    except Exception as e:
        return jsonify({'success': False, 'error': _bilt_conn_err(e)})


# BILT 物体検知
@app.route('/api/bilt/detection/settings', methods=['GET', 'POST'])
def bilt_detection_settings_proxy():
    try:
        if request.method == 'POST':
            return jsonify(bilt_client.update_detection_settings(request.json))
        return jsonify(bilt_client.get_detection_settings())
    except Exception as e:
        return jsonify({'success': False, 'error': _bilt_conn_err(e)})


@app.route('/api/bilt/detection/start', methods=['POST'])
def bilt_start_detection():
    try:
        return jsonify(bilt_client.start_detection())
    except Exception as e:
        return jsonify({'success': False, 'error': _bilt_conn_err(e)})


@app.route('/api/bilt/detection/stop', methods=['POST'])
def bilt_stop_detection():
    try:
        return jsonify(bilt_client.stop_detection())
    except Exception as e:
        return jsonify({'success': False, 'error': _bilt_conn_err(e)})


@app.route('/api/bilt/detection/status')
def bilt_detection_status_proxy():
    try:
        return jsonify(bilt_client.get_detection_status())
    except Exception as e:
        return jsonify({'active': False, 'error': _bilt_conn_err(e)})


@app.route('/api/bilt/detection/stats')
def bilt_detection_stats_proxy():
    try:
        return jsonify(bilt_client.get_detection_stats())
    except Exception as e:
        return jsonify({'fps': 0, 'error': _bilt_conn_err(e)})


@app.route('/api/bilt/detections/latest')
def bilt_detections_latest():
    try:
        return jsonify(bilt_client.get_latest_detections())
    except Exception as e:
        return jsonify({'detections': [], 'error': _bilt_conn_err(e)})


def generate_bilt_detection_frames():
    """BILT MJPEGストリームを生成します."""
    import time
    try:
        while True:
            try:
                frame_bytes = bilt_client.get_latest_frame()
                if frame_bytes:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    import numpy as np
                    blank = cv2.imencode('.jpg',
                        cv2.putText(
                            __import__('numpy').zeros((480, 640, 3), dtype=__import__('numpy').uint8),
                            'Waiting for BILT Service...', (80, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2
                        ))[1].tobytes()
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + blank + b'\r\n')
            except GeneratorExit:
                return
            except Exception as e:
                logger.error(f"app.py - BILT frame gen error: {e}")
            time.sleep(0.067)
    except GeneratorExit:
        pass


@app.route('/bilt/detection/video_feed')
def bilt_detection_video_feed():
    return Response(generate_bilt_detection_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# BILT トレーニング
@app.route('/api/bilt/train', methods=['POST'])
def bilt_start_training():
    if not current_project:
        return jsonify({'error': 'プロジェクトがロードされていません'}), 400
    cfg = request.json or {}
    cfg['project_path'] = current_project.path
    try:
        return jsonify(bilt_client.start_training(cfg))
    except Exception as e:
        return jsonify({'error': _bilt_conn_err(e)}), 500


@app.route('/api/bilt/train/stop', methods=['POST'])
def bilt_stop_training():
    try:
        return jsonify(bilt_client.stop_training())
    except Exception as e:
        return jsonify({'success': False, 'error': _bilt_conn_err(e)})


@app.route('/api/bilt/train/status')
def bilt_training_status_proxy():
    try:
        return jsonify(bilt_client.get_training_status())
    except Exception as e:
        return jsonify({'active': False, 'phase': 'idle', 'error': _bilt_conn_err(e)})


@app.route('/api/bilt/project/models')
def bilt_project_models():
    if not current_project:
        return jsonify({'models': [], 'has_model': False})
    try:
        return jsonify(bilt_client.list_project_models(current_project.path))
    except Exception as e:
        return jsonify({'models': [], 'has_model': False, 'error': _bilt_conn_err(e)})


# BILT 再ラベリング
@app.route('/api/bilt/relabel/models')
def bilt_relabel_models():
    if not current_project:
        return jsonify({'models': []})
    try:
        return jsonify(bilt_client.get_relabel_models(current_project.path))
    except Exception as e:
        return jsonify({'models': [], 'error': str(e)})


@app.route('/api/bilt/relabel/start', methods=['POST'])
def bilt_relabel_start():
    if not current_project:
        return jsonify({'error': 'プロジェクトがロードされていません'}), 400
    cfg = request.json or {}
    cfg['project_path'] = current_project.path
    try:
        return jsonify(bilt_client.start_relabel(cfg))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# BILT テスト推論
@app.route('/api/bilt/test/image', methods=['POST'])
def bilt_test_image():
    """アップロード画像を BILT モデルで推論し、アノテーション済み画像と検出結果を返す。"""
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'image フィールドが必要です'}), 400
    f = request.files['image']
    image_bytes = f.read()
    model_path = request.form.get('model_path', '')
    conf = float(request.form.get('conf', 0.25))
    iou = float(request.form.get('iou', 0.45))
    try:
        result = bilt_client.test_image(
            image_bytes=image_bytes,
            filename=f.filename or 'image.jpg',
            model_path=model_path,
            conf=conf,
            iou=iou,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# エラーハンドラ
# ─────────────────────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(_):
    return jsonify({'error': '見つかりません'}), 404


@app.errorhandler(500)
def server_error(e):
    logger.error(f"サーバーエラー: {e}")
    return jsonify({'error': '内部サーバーエラー'}), 500


# ─────────────────────────────────────────────────────────────────────────────
# アラートフィード  (WorkflowEngineからのアラートを受け取って、ブラウザで表示するためのシンプルなエンドポイント)
# ─────────────────────────────────────────────────────────────────────────────

from collections import deque
_alert_feed: deque = deque(maxlen=200)   # メモリに最後の200個のアラートを保持
_alert_feed_lock = threading.Lock()


@app.route('/api/alert_feed', methods=['POST'])
def alert_feed_receive():
    """アラートフィード: WorkflowEngineからのアラートを受け取ります."""
    try:
        data = request.get_json(force=True, silent=True) or {}
    except Exception:
        data = {}
    data.setdefault('timestamp', datetime.utcnow().isoformat())
    data.setdefault('source', 'unknown')
    with _alert_feed_lock:
        _alert_feed.append(data)
    return jsonify({'ok': True})


@app.route('/api/alert_feed', methods=['GET'])
def alert_feed_get():
    """全てのバッファされたアラートを返します (最新のものから).  クライエントから新しいアラートを取得するには ?since=N を指定します."""
    since = int(request.args.get('since', 0))
    with _alert_feed_lock:
        items = list(_alert_feed)
    return jsonify({'alerts': items[since:], 'total': len(items)})


@app.route('/alert_viewer')
def alert_viewer():
    return render_template('alert_viewer.html')


# ─────────────────────────────────────────────────────────────────────────────
# メインアプリ – Flask + SocketIO サーバーを起動します
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(projects_dir, exist_ok=True)
    HOST, PORT = '127.0.0.1', 5000
    print(f"全アプリを http://{HOST}:{PORT} で起動しています")
    print(f"BILT serviceは http://127.0.0.1:5002 で実行中です")
    if check_bilt_service():
        print("BILT サービス: オンライン")
    else:
        print("注意: BILT サービスが利用できません - bilt_service.py を先に起動してください")
    # Windows上のソケットエラー"client closed tab"を消します
    class _QuietWsgiLog:
        _IGNORE = ('ConnectionAbortedError', 'WinError 10053', 'BrokenPipeError',
                   'ConnectionResetError', 'WinError 10054')

        def write(self, msg):
            if not any(s in msg for s in self._IGNORE):
                sys.stderr.write(msg)

        def flush(self):
            sys.stderr.flush()

    listener = eventlet.listen((HOST, PORT))
    eventlet.wsgi.server(listener, app, log=_QuietWsgiLog())
