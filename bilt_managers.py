# Copyright (C) 2026 Rikiza89
# Licensed under the Apache License, Version 2.0
"""
BILT_Workflow_JA マネージャー - カメラ、チェーン物体検知、および画像管理。
bilt_service.pyが使用します。
"""

import cv2
import numpy as np
import os
import sys
import json
import logging
import time
from datetime import datetime

# ── Unicode / 日本語テキスト描画ヘルパー (PIL) ───────────────────────────────────

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf",
    "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]

def _load_pil_font(size: int = 18):
    """サイズ指定でPILフォントを返す。利用可能な日本語フォントを優先使用。"""
    try:
        from PIL import ImageFont
        for path in _FONT_CANDIDATES:
            if os.path.exists(path):
                return ImageFont.truetype(path, size)
        return ImageFont.load_default()
    except Exception:
        return None


def put_text_unicode(frame: np.ndarray, text: str, xy: tuple,
                     font_size: int = 18, color=(255, 255, 255),
                     bg_color=None) -> np.ndarray:
    """PILを使いUnicodeテキスト（日本語含む）をBGR ndarrayに描画する。"""
    try:
        from PIL import Image, ImageDraw
        font = _load_pil_font(font_size)
        if font is None:
            cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
                        font_size / 30, color, 1, cv2.LINE_AA)
            return frame
        # BGR → RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        x, y = xy
        pil_color = (color[2], color[1], color[0])  # BGR→RGB
        if bg_color is not None:
            try:
                bbox = draw.textbbox((x, y), text, font=font)
                pad = 2
                draw.rectangle([bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad],
                                fill=(bg_color[2], bg_color[1], bg_color[0]))
            except Exception:
                pass
        draw.text((x, y), text, fill=pil_color, font=font)
        frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return frame
    except Exception:
        # PILが使えない場合はフォールバック（Kanjiは?になるが動作は維持）
        cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
                    font_size / 30, color, 1, cv2.LINE_AA)
        return frame

# ── picamera2 バックエンド (Raspberry Pi カメラモジュール) ───────────────────────────

PICAMERA2_INDEX = 'picamera2'   # これをcamera_indexとして使用してPi cameraを選択します


class _PiCamera2Backend:
    """picamera2をcv2.VideoCapture-インターフェースでラップしてから使用できるようにします。"""

    def __init__(self, width: int, height: int, fps: float):
        from picamera2 import Picamera2  # type: ignore
        self._cam = Picamera2()
        cfg = self._cam.create_video_configuration(
            main={'size': (width, height), 'format': 'BGR888'},
            controls={'FrameRate': float(fps)},
        )
        self._cam.configure(cfg)
        self._cam.start()
        self._opened = True
        self._width  = width
        self._height = height
        self._fps    = fps

    # ── cv2-的なインターフェース ────────────────────────────────────────────────────

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        try:
            frame = self._cam.capture_array()
            return True, frame
        except Exception:
            return False, None

    def release(self):
        if self._opened:
            try:
                self._cam.stop()
            except Exception:
                pass
            self._opened = False

    def set(self, prop, val):
        pass  
    
    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        if prop == cv2.CAP_PROP_FPS:
            return float(self._fps)
        return 0.0

    def getBackendName(self) -> str:
        return 'picamera2'

logger = logging.getLogger(__name__)

# 承認されたスキップの後の数秒間、新しいスキップは抑制されます
SKIP_GRACE_PERIOD = 3.0


class EnhancedCameraManager:
    def __init__(self):
        self.cap = None
        self.lock = __import__('threading').Lock()
        self.current_index = None
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0
        self.color_mode = 'bgr'
        self.flip_horizontal = False

    def get_available_cameras(self):
        cameras = []
        camera_names = {}

        # ── Windows: DirectShow probe (same backend as initialize_camera) + PowerShell names ──
        if sys.platform == 'win32':
            # Probe with DirectShow first to discover real, working cv2 indices.
            # Then zip PowerShell friendly-names by position — both enumerate in the same order.
            working_indices: list = []
            for i in range(5):
                try:
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                    if cap.isOpened():
                        working_indices.append(i)
                    cap.release()
                except Exception:
                    pass
            ps_names: list = []
            try:
                import subprocess as _sp
                result = _sp.run(
                    ['powershell', '-Command',
                     'Get-PnpDevice -Class Camera | '
                     'Select-Object FriendlyName, Status | ConvertTo-Json'],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    devices = json.loads(result.stdout)
                    if not isinstance(devices, list):
                        devices = [devices]
                    ps_names = [d.get('FriendlyName', '') for d in devices if d.get('Status') == 'OK']
            except Exception:
                pass
            for pos, idx in enumerate(working_indices):
                name = ps_names[pos] if pos < len(ps_names) and ps_names[pos] else f'Camera {idx}'
                # Open once more to read resolution/fps
                try:
                    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                    if cap.isOpened():
                        cameras.append({
                            'index':  idx,
                            'width':  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                            'fps':    cap.get(cv2.CAP_PROP_FPS),
                            'name':   name,
                        })
                        cap.release()
                    else:
                        cap.release()
                        cameras.append({'index': idx, 'width': 640, 'height': 480, 'fps': 30.0, 'name': name})
                except Exception:
                    cameras.append({'index': idx, 'width': 640, 'height': 480, 'fps': 30.0, 'name': name})
            return cameras

        # ── Linux: /dev/video* ノードをプローブする ────────────────────────────────────
        elif sys.platform.startswith('linux'):
            import glob as _glob
            nodes = sorted(_glob.glob('/dev/video*'))
            # /dev/video0、/dev/video1、…から数値インデックスを抽出します。
            probe_indices = []
            for node in nodes:
                try:
                    probe_indices.append(int(node.replace('/dev/video', '')))
                except ValueError:
                    pass
            if not probe_indices:
                probe_indices = list(range(4))
            for i in probe_indices:
                cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
                if cap.isOpened():
                    try:
                        cameras.append({
                            'index': i,
                            'width':  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                            'fps':    cap.get(cv2.CAP_PROP_FPS),
                            'name':   camera_names.get(i, f'USB Camera {i} (/dev/video{i})'),
                        })
                    except Exception:
                        cameras.append({'index': i, 'width': 640, 'height': 480,
                                        'fps': 30.0, 'name': f'USB Camera {i}'})
                    finally:
                        cap.release()
            # picamera2 がインストールされている場合は、Pi カメラ モジュールを追加します
            try:
                import picamera2  # noqa: F401
                cameras.append({
                    'index':  PICAMERA2_INDEX,
                    'width':  1920,
                    'height': 1080,
                    'fps':    30.0,
                    'name':   'Raspberry Pi Camera (picamera2)',
                })
            except ImportError:
                pass
            return cameras

        # ── macOS / フォールバック ──────────────────────────────────────────────────
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                try:
                    cameras.append({
                        'index':  i,
                        'width':  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        'fps':    cap.get(cv2.CAP_PROP_FPS),
                        'name':   camera_names.get(i, f'Camera {i}'),
                    })
                except Exception:
                    cameras.append({'index': i, 'width': 640, 'height': 480,
                                    'fps': 30.0, 'name': f'Camera {i}'})
                finally:
                    cap.release()
        return cameras

    def _detect_color_issue(self, frame):
        if frame is None or len(frame.shape) != 3:
            return None
        b_avg, g_avg, r_avg = frame[:, :, 0].mean(), frame[:, :, 1].mean(), frame[:, :, 2].mean()
        if b_avg > r_avg * 1.5 and b_avg > 100:
            return 'yuv2bgr'
        elif r_avg > b_avg * 1.5 and r_avg > 100:
            return 'rgb2bgr'
        return None

    def initialize_camera(self, index, config):
        with self.lock:
            try:
                if self.cap:
                    self.cap.release()
                    self.cap = None

                w   = config['DEFAULT_CAMERA_WIDTH']
                h   = config['DEFAULT_CAMERA_HEIGHT']
                fps = config['DEFAULT_CAMERA_FPS']
                self.flip_horizontal = bool(config.get('CAMERA_FLIP_HORIZONTAL', False))

                # ── picamera2 (Raspberry Pi カメラモジュール) ────────────────────
                if index == PICAMERA2_INDEX:
                    try:
                        self.cap = _PiCamera2Backend(w, h, fps)
                        self.current_index = index
                        self.color_mode = 'bgr'
                        logger.info('Camera: using picamera2 backend')
                        return True
                    except Exception as e:
                        logger.error(f'picamera2 init error: {e}')
                        return False

                # ── OpenCV バックエンド (OSによって優先度が異なります) ──────────────────
                if sys.platform == 'win32':
                    backends = [
                        (cv2.CAP_DSHOW, 'DirectShow'),
                        (cv2.CAP_MSMF,  'Media Foundation'),
                        (cv2.CAP_ANY,   'Default'),
                    ]
                elif sys.platform.startswith('linux'):
                    backends = [
                        (cv2.CAP_V4L2, 'V4L2'),
                        (cv2.CAP_ANY,  'Default'),
                    ]
                else:
                    backends = [
                        (cv2.CAP_ANY, 'Default'),
                    ]

                for backend, name in backends:
                    try:
                        test_cap = cv2.VideoCapture(index, backend)
                        if test_cap.isOpened():
                            self.cap = test_cap
                            logger.info(f'Camera {index}: using {name} backend')
                            break
                        test_cap.release()
                    except Exception:
                        continue

                if not self.cap or not self.cap.isOpened():
                    return False

                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                self.cap.set(cv2.CAP_PROP_FPS,          fps)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   config['FRAME_BUFFER_SIZE'])
                self.current_index = index
                self.color_mode = 'bgr'
                return True
            except Exception as e:
                logger.error(f'カメラ初期化エラー: {e}')
                if self.cap:
                    self.cap.release()
                    self.cap = None
                return False

    def get_frame(self):
        with self.lock:
            if self.cap is None or not self.cap.isOpened():
                return None, 0
            ret, frame = self.cap.read()
            if ret and frame is not None:
                if self.color_mode == 'yuv2bgr':
                    try:
                        frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
                    except Exception:
                        pass
                elif self.color_mode == 'rgb2bgr':
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if self.flip_horizontal:
                    frame = cv2.flip(frame, 1)

                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.fps = round(self.frame_count / (current_time - self.last_fps_time), 1)
                    self.frame_count = 0
                    self.last_fps_time = current_time
                return frame, self.fps
            return None, 0

    def get_camera_info(self):
        with self.lock:
            if not self.cap or not self.cap.isOpened():
                return None
            fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            return {
                'index': self.current_index,
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.cap.get(cv2.CAP_PROP_FPS),
                'backend': self.cap.getBackendName(),
                'fourcc': fourcc_str,
                'color_mode': self.color_mode,
            }

    def release(self):
        with self.lock:
            if self.cap:
                self.cap.release()
                self.cap = None
                self.current_index = None
                self.color_mode = 'bgr'


class ChainDetectionManager:
    """順次ステップベースの検出ワークフロー。"""

    @staticmethod
    def make_initial_state():
        """新しいチェーン状態の辞書を返します。"""
        return {
            'active': False,
            'current_step': 0,
            'step_start_time': None,
            'completed_cycles': 0,
            'failed_steps': 0,
            'step_history': [],
            'current_detections': {},
            'last_step_result': None,
            'cycle_pause': False,
            'cycle_pause_start': None,
            'skip_pause': False,
            'skip_pause_step': None,
            'skipped_class_name': None,
            # 最後にスキップが確認されたときのタイムスタンプ。猶予期間に使用されます。
            'skip_ack_time': 0,
        }

    @staticmethod
    def initialize_chain(chain_state):
        chain_state.update({
            'active': True,
            'current_step': 0,
            'step_start_time': time.time(),
            'completed_cycles': 0,
            'failed_steps': 0,
            'step_history': [],
            'current_detections': {},
            'last_step_result': None,
            'cycle_pause': False,
            'cycle_pause_start': None,
            'skip_pause': False,
            'skip_pause_step': None,
            'skipped_class_name': None,
            'skip_ack_time': 0,
        })

    @staticmethod
    def process_chain_detection(detections, detection_settings, chain_state, *_args):
        """チェーンモードで検出を処理します。

        (検出、chain_result_dict | なし) を返します。
        """
        if not detection_settings['chain_mode'] or not chain_state['active']:
            return detections, None

        if not detection_settings['chain_steps']:
            return detections, {'error': 'チェーンステップが設定されていません。'}

        current_time = time.time()

        # このフレームのクラスごとのカウントを構築
        step_detections = {}
        for d in detections:
            cls = d['class_name']
            step_detections[cls] = step_detections.get(cls, 0) + 1

        chain_state['current_detections'] = step_detections

        # ── スキップ一時停止: ユーザーの確認を待つ ──────────────────────
        if chain_state.get('skip_pause', False):
            return detections, {
                'step': chain_state.get('skip_pause_step', chain_state['current_step']),
                'step_name': 'SKIP 検知されました – 了解ボタンを押してください',
                'detected': step_detections,
                'skip_pause': True,
                'timestamp': current_time,
            }

        # ── サイクル一時停止: 完了シーケンス間の休止 ──────────────────
        if chain_state.get('cycle_pause', False):
            pause_elapsed = current_time - chain_state['cycle_pause_start']
            pause_remaining = detection_settings['chain_pause_time'] - pause_elapsed
            if pause_remaining <= 0:
                chain_state['cycle_pause'] = False
                chain_state['cycle_pause_start'] = None
                chain_state['step_start_time'] = current_time
            else:
                return detections, {
                    'step': -1,
                    'step_name': 'Cycle Pause',
                    'detected': step_detections,
                    'remaining_pause': pause_remaining,
                    'timestamp': current_time,
                }

        # ── サイクル完了チェック ─────────────────────────────────────────
        if chain_state['current_step'] >= len(detection_settings['chain_steps']):
            chain_state['current_step'] = 0
            chain_state['completed_cycles'] += 1
            chain_state['cycle_pause'] = True
            chain_state['cycle_pause_start'] = current_time
            return detections, {
                'step': -1,
                'step_name': 'Cycle Pause',
                'detected': step_detections,
                'remaining_pause': detection_settings['chain_pause_time'],
                'timestamp': current_time,
            }

        current_step_idx = chain_state['current_step']
        current_step_config = detection_settings['chain_steps'][current_step_idx]

        # ── スキップ検出 (猶予期間外のみ) ────────────────────
        skip_grace_ok = (current_time - chain_state.get('skip_ack_time', 0)) > SKIP_GRACE_PERIOD
        if skip_grace_ok:
            is_skipped, skipped_to_step = ChainDetectionManager.check_for_skip(
                step_detections, current_step_idx, detection_settings
            )
            if is_skipped:
                chain_state['skip_pause'] = True
                chain_state['skip_pause_step'] = current_step_idx
                return detections, {
                    'step': current_step_idx,
                    'step_name': current_step_config['name'],
                    'detected': step_detections,
                    'skip_pause': True,
                    'skipped_to_step': skipped_to_step,
                    'timestamp': current_time,
                }

        # ── ステップ完了チェック ─────────────────────────────────────────
        required = current_step_config.get('classes', {})
        step_completed = bool(required) and all(
            step_detections.get(cls, 0) >= cnt for cls, cnt in required.items()
        )
        step_timeout = (
            chain_state['step_start_time'] is not None and
            current_time - chain_state['step_start_time'] > detection_settings['chain_timeout']
        )

        result = {
            'step': current_step_idx,
            'step_name': current_step_config['name'],
            'detected': step_detections,
            'required': required,
            'completed': step_completed,
            'timeout': step_timeout and not step_completed,
            'timestamp': current_time,
        }

        if step_completed:
            chain_state['step_history'].append({
                'step': current_step_idx,
                'name': current_step_config['name'],
                'result': 'success',
                'timestamp': current_time,
            })
            chain_state['current_step'] += 1
            chain_state['step_start_time'] = current_time
            chain_state['last_step_result'] = 'success'

        elif step_timeout and detection_settings['chain_auto_advance']:
            chain_state['step_history'].append({
                'step': current_step_idx,
                'name': current_step_config['name'],
                'result': 'timeout',
                'timestamp': current_time,
            })
            chain_state['failed_steps'] += 1
            chain_state['current_step'] += 1
            chain_state['step_start_time'] = current_time
            chain_state['last_step_result'] = 'timeout'
            result['timeout'] = True

        return detections, result

    @staticmethod
    def check_for_skip(step_detections, current_step_index, detection_settings):
        """未来のステップのユニーククラスが検出された場合に (True, 未来のステップインデックス) を返します."""
        steps = detection_settings['chain_steps']
        if not steps:
            return False, None
        current_classes = set(steps[current_step_index].get('classes', {}).keys())
        for future_idx in range(current_step_index + 1, len(steps)):
            for cls, required_count in steps[future_idx].get('classes', {}).items():
                if cls in current_classes:
                    continue
                if step_detections.get(cls, 0) >= required_count:
                    return True, future_idx
        return False, None

    @staticmethod
    def reset_chain(chain_state):
        chain_state.update({
            'current_step': 0,
            'step_start_time': time.time(),
            'step_history': [],
            'current_detections': {},
            'last_step_result': None,
            'cycle_pause': False,
            'cycle_pause_start': None,
            'skip_pause': False,
            'skip_pause_step': None,
            'skipped_class_name': None,
        })

    @staticmethod
    def acknowledge_skip(chain_state):
        """スキップ一時停止をクリアし、同じスキップが再トリガーされないように猶予期間を開始します."""
        chain_state['skip_pause'] = False
        chain_state['skip_pause_step'] = None
        chain_state['skipped_class_name'] = None
        chain_state['last_step_result'] = None
        chain_state['step_start_time'] = time.time()
        chain_state['skip_ack_time'] = time.time()

    @staticmethod
    def get_chain_status(detection_settings, chain_state):
        if not detection_settings['chain_mode']:
            return {'active': False}

        current_step_config = None
        steps = detection_settings['chain_steps']
        if steps and chain_state['current_step'] < len(steps):
            current_step_config = steps[chain_state['current_step']]

        remaining_time = 0
        if chain_state['step_start_time'] is not None:
            remaining_time = max(
                0,
                detection_settings['chain_timeout'] - (time.time() - chain_state['step_start_time'])
            )

        pause_remaining = 0
        if chain_state['cycle_pause'] and chain_state['cycle_pause_start']:
            pause_remaining = max(
                0,
                detection_settings['chain_pause_time'] - (time.time() - chain_state['cycle_pause_start'])
            )

        return {
            'active': chain_state['active'],
            'current_step': chain_state['current_step'],
            'total_steps': len(steps),
            'current_step_config': current_step_config,
            'completed_cycles': chain_state['completed_cycles'],
            'failed_steps': chain_state['failed_steps'],
            'current_detections': chain_state['current_detections'],
            'remaining_time': remaining_time,
            'last_result': chain_state['last_step_result'],
            'step_history': chain_state['step_history'][-10:],
            'cycle_pause': chain_state['cycle_pause'],
            'pause_remaining': pause_remaining,
            'skip_pause': chain_state.get('skip_pause', False),
            'skipped_class_name': chain_state.get('skipped_class_name'),
        }


class DetectionProcessor:
    COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
    ]

    @staticmethod
    def process_detections(results, frame, detection_settings, object_counters,
                           detection_stats, chain_state, counter_triggered, *_args):
        detections = []
        if not results or len(results) == 0:
            return detections, frame

        result = results[0]
        has_masks = hasattr(result, 'masks') and result.masks is not None
        has_obb = hasattr(result, 'obb') and result.obb is not None
        has_keypoints = hasattr(result, 'keypoints') and result.keypoints is not None

        if has_masks:
            for box, mask in zip(result.boxes, result.masks.xy):
                try:
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    if detection_settings['classes'] and class_id not in detection_settings['classes']:
                        continue
                    detections.append({
                        'bbox': box.xyxy[0].cpu().numpy().tolist(),
                        'polygon': [[float(x), float(y)] for x, y in mask],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': result.names[class_id],
                        'type': 'segment',
                    })
                except Exception:
                    pass

        elif has_obb:
            for obb in result.obb:
                try:
                    confidence = float(obb.conf[0].cpu().numpy())
                    class_id = int(obb.cls[0].cpu().numpy())
                    if detection_settings['classes'] and class_id not in detection_settings['classes']:
                        continue
                    detections.append({
                        'corners': obb.xyxyxyxy[0].cpu().numpy().tolist(),
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': result.names[class_id],
                        'type': 'obb',
                    })
                except Exception:
                    pass

        elif has_keypoints:
            boxes = result.boxes if hasattr(result, 'boxes') else None
            for i, kp in enumerate(result.keypoints):
                try:
                    class_id = int(boxes.cls[i].cpu().numpy()) if boxes else 0
                    confidence = float(boxes.conf[i].cpu().numpy()) if boxes else 1.0
                    detections.append({
                        'bbox': boxes.xyxy[i].cpu().numpy().tolist() if boxes else None,
                        'keypoints': kp.xy[0].cpu().numpy().tolist(),
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': result.names.get(class_id, 'person'),
                        'type': 'pose',
                    })
                except Exception:
                    pass

        else:
            if result.boxes:
                for box in result.boxes:
                    try:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        if detection_settings['classes'] and class_id not in detection_settings['classes']:
                            continue
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': result.names[class_id],
                            'type': 'detect',
                        })
                    except Exception:
                        pass

        # フレームごとのクラスカウント
        frame_detections = {}
        for det in detections:
            cls = det['class_name']
            frame_detections[cls] = frame_detections.get(cls, 0) + 1

        # チェーンモード処理
        chain_result = None
        if detection_settings['chain_mode']:
            detections, chain_result = ChainDetectionManager.process_chain_detection(
                detections, detection_settings, chain_state
            )

        # カウンターモード (チェーンモードと相互排他的)
        if detection_settings['counter_mode'] and not detection_settings['chain_mode']:
            for cls, count in frame_detections.items():
                if cls not in counter_triggered:
                    object_counters[cls] = object_counters.get(cls, 0) + count
                    counter_triggered[cls] = True

        detection_stats['total_detections'] += len(detections)
        detection_stats['last_detection_time'] = datetime.now()

        annotated = DetectionProcessor.draw_detections(
            frame.copy(), detections, chain_result, detection_settings, chain_state, detection_stats
        )
        return detections, annotated

    @staticmethod
    def draw_detections(frame, detections, chain_result, detection_settings, chain_state, detection_stats):
        colors = DetectionProcessor.COLORS
        for det in detections:
            cls = det['class_name']
            conf = det['confidence']
            color = colors[det['class_id'] % len(colors)]

            if det.get('type') == 'segment' and 'polygon' in det:
                poly = np.array(det['polygon'], dtype=np.int32)
                cv2.polylines(frame, [poly], True, color, 2)
                if len(poly) > 0:
                    x, y = poly[0]
                    # ラベルはJS側のcanvasオーバーレイで表示（Kanji対応）

            elif det.get('type') == 'obb' and 'corners' in det:
                corners = np.array(det['corners'], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [corners], True, color, 2)
                # ラベルはJS側のcanvasオーバーレイで表示（Kanji対応）

            elif det.get('type') == 'pose' and 'keypoints' in det:
                kps = np.array(det['keypoints'], dtype=np.int32)
                if det.get('bbox'):
                    x1, y1, x2, y2 = det['bbox']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                skeleton = [
                    (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15),
                    (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12),
                ]
                for a, b in skeleton:
                    if a < len(kps) and b < len(kps):
                        cv2.line(frame, tuple(kps[a]), tuple(kps[b]), color, 2)
                for x, y in kps:
                    cv2.circle(frame, (x, y), 4, color, -1)

            else:
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # ラベルはJS側のcanvasオーバーレイで表示（Kanji対応）

        if detection_settings['chain_mode']:
            DetectionProcessor.draw_chain_overlay(frame, chain_state, detection_settings)
        else:
            put_text_unicode(frame, f"FPS: {detection_stats['fps']}", (10, 12),
                             font_size=18, color=(0, 255, 0))
            put_text_unicode(frame, f"Det: {len(detections)}", (10, 34),
                             font_size=18, color=(0, 255, 0))

        return frame

    @staticmethod
    def draw_chain_overlay(frame, chain_state, detection_settings):
        steps = detection_settings['chain_steps']
        if not steps:
            return
        current_step = chain_state['current_step']
        total_steps = len(steps)
        bar_w, bar_h = 400, 30
        bar_x = (frame.shape[1] - bar_w) // 2
        bar_y = 20
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)

        if chain_state.get('skip_pause', False):
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (0, 0, 200), -1)
            put_text_unicode(frame, "スキップ! – 了解ボタンを押してください",
                             (bar_x + 10, bar_y + 6), font_size=16, color=(255, 255, 255))
        elif chain_state.get('cycle_pause', False):
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (0, 165, 255), -1)
            put_text_unicode(frame, "CYCLE PAUSE", (bar_x + 130, bar_y + 6),
                             font_size=16, color=(255, 255, 255))
        else:
            progress = current_step / total_steps if total_steps > 0 else 0
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + int(bar_w * progress), bar_y + bar_h), (0, 200, 0), -1)
            label = f"Step {min(current_step + 1, total_steps)}/{total_steps}"
            put_text_unicode(frame, label, (bar_x + 10, bar_y + 6),
                             font_size=16, color=(255, 255, 255))


class ImageManager:
    @staticmethod
    def save_detection_image(frame, detections, config, prefix="detection"):
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{prefix}_{ts}.jpg"
            filepath = os.path.join(config['SAVED_IMAGES_DIR'], filename)
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, config['SAVE_IMAGE_QUALITY']])
            return filename
        except Exception as e:
            logger.error(f"画像保存エラー: {e}")
            return None

    @staticmethod
    def save_dataset_image(frame, detections, project_folder, config):
        if not project_folder:
            return None
        try:
            from werkzeug.utils import secure_filename
            project_path = os.path.join(config['PROJECTS_DIR'], secure_filename(project_folder))
            images_path = os.path.join(project_path, 'train', 'images')
            labels_path = os.path.join(project_path, 'train', 'labels')
            os.makedirs(images_path, exist_ok=True)
            os.makedirs(labels_path, exist_ok=True)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            img_file = f"img_{ts}.jpg"
            lbl_file = f"img_{ts}.txt"

            cv2.imwrite(os.path.join(images_path, img_file), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, config['SAVE_IMAGE_QUALITY']])

            h, w = frame.shape[:2]
            with open(os.path.join(labels_path, lbl_file), 'w') as f:
                for det in detections:
                    if det.get('type') == 'segment' and 'polygon' in det:
                        coords = ' '.join(f"{x/w:.6f} {y/h:.6f}" for x, y in det['polygon'])
                        f.write(f"{det['class_id']} {coords}\n")
                    elif det.get('type') == 'obb' and 'corners' in det:
                        coords = ' '.join(f"{x/w:.6f} {y/h:.6f}" for x, y in det['corners'])
                        f.write(f"{det['class_id']} {coords}\n")
                    else:
                        x1, y1, x2, y2 = det['bbox']
                        cx = (x1 + x2) / 2 / w
                        cy = (y1 + y2) / 2 / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        f.write(f"{det['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            return img_file
        except Exception as e:
            logger.error(f"データセット画像保存エラー: {e}")
            return None


class RGBBalancer:
    def __init__(self):
        self.red_gain = 1.0
        self.green_gain = 1.0
        self.blue_gain = 1.0

    def set_gains(self, red, green, blue):
        self.red_gain = red / 128.0
        self.green_gain = green / 128.0
        self.blue_gain = blue / 128.0

    def apply(self, frame):
        if frame is None:
            return frame
        b, g, r = cv2.split(frame)
        r = np.clip(r * self.red_gain, 0, 255).astype(np.uint8)
        g = np.clip(g * self.green_gain, 0, 255).astype(np.uint8)
        b = np.clip(b * self.blue_gain, 0, 255).astype(np.uint8)
        return cv2.merge([b, g, r])
