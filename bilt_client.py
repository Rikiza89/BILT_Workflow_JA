# Copyright (C) 2026 Rikiza89
# Licensed under the Apache License, Version 2.0
"""
BILT クライエント - BILT トレーニングと物体検知の API をカバーします。
port 5002 で実行されている bilt_service.py と通信します。
"""

import requests
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

BILT_SERVICE_URL = 'http://127.0.0.1:5002'


def _conn_err_msg(e: Exception) -> str:
    msg = str(e)
    if any(k in msg for k in ('Connection refused', 'WSAECONNREFUSED',
                               'NewConnectionError', 'Max retries')):
        return 'BILTサービスが起動していません。bilt_service.py を先に起動してください。'
    return msg


class BILTClient:
    """BILT バックエンドサービス用 HTTP クライエント"""

    def __init__(self, base_url: str = BILT_SERVICE_URL):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})

    def _get(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        return self._request('GET', endpoint, **kwargs)

    def _post(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        return self._request('POST', endpoint, **kwargs)

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        url = f'{self.base_url}{endpoint}'
        timeout = kwargs.pop('timeout', 10)
        try:
            resp = self.session.request(method, url, timeout=timeout, **kwargs)
            if not resp.ok:
                # Extract the actual error from the JSON body when available so
                # the caller sees the Python exception, not the HTTP wrapper.
                try:
                    body = resp.json()
                    error_msg = body.get('error') or body.get('message') or resp.text
                except Exception:
                    error_msg = resp.text or f'HTTP {resp.status_code}'
                logger.error(f'BILT {method} {endpoint} → {resp.status_code}: {error_msg}')
                return {'success': False, 'error': error_msg}
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f'BILT リクエスト {method} {endpoint} 失敗: {e}')
            return {'success': False, 'error': _conn_err_msg(e)}

    # ── ヘルスチェック ──────────────────────────────────────────────────────────

    def health_check(self) -> Dict[str, Any]:
        return self._get('/health')

    # ── モデル管理 ──────────────────────────────────────────────────────────────

    def get_models(self) -> Dict[str, Any]:
        return self._get('/api/bilt/models')

    def get_model_params(self, name: str) -> Dict[str, Any]:
        return self._get(f'/api/bilt/models/params', params={'name': name})

    def load_model(self, model_name: str) -> Dict[str, Any]:
        return self._post('/api/bilt/model/load', json={'model_name': model_name})

    def get_model_info(self) -> Dict[str, Any]:
        return self._get('/api/bilt/model/info')

    # ── カメラ ───────────────────────────────────────────────────────────────

    def get_cameras(self) -> Dict[str, Any]:
        return self._get('/api/bilt/cameras')

    def select_camera(self, index: int) -> Dict[str, Any]:
        return self._post('/api/bilt/camera/select', json={'camera_index': index})

    # ── 物体検知 ────────────────────────────────────────────────────────────

    def get_detection_settings(self) -> Dict[str, Any]:
        return self._get('/api/bilt/detection/settings')

    def update_detection_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        return self._post('/api/bilt/detection/settings', json=settings)

    def start_detection(self) -> Dict[str, Any]:
        return self._post('/api/bilt/detection/start')

    def stop_detection(self) -> Dict[str, Any]:
        return self._post('/api/bilt/detection/stop')

    def get_detection_status(self) -> Dict[str, Any]:
        return self._get('/api/bilt/detection/status')

    def get_detection_stats(self) -> Dict[str, Any]:
        return self._get('/api/bilt/detection/stats')

    def get_latest_detections(self) -> Dict[str, Any]:
        return self._get('/api/bilt/detections/latest')

    def get_latest_frame(self) -> Optional[bytes]:
        try:
            resp = self.session.get(f'{self.base_url}/api/bilt/frame/latest', timeout=5)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            logger.error(f'BILT frame fetch error: {e}')
            return None

    # ── カウンター ───────────────────────────────────────────────────────────────

    def get_counters(self) -> Dict[str, Any]:
        return self._get('/api/bilt/counters')

    def reset_counters(self) -> Dict[str, Any]:
        return self._post('/api/bilt/counters/reset')

    # ── チェーン検出 ─────────────────────────────────────────────────────────────

    def get_chain_status(self) -> Dict[str, Any]:
        return self._get('/api/bilt/chain/status')

    def chain_control(self, action: str) -> Dict[str, Any]:
        return self._post('/api/bilt/chain/control', json={'action': action})

    def get_chain_config(self) -> Dict[str, Any]:
        return self._get('/api/bilt/chain/config')

    def update_chain_config(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self._post('/api/bilt/chain/config', json=data)

    def acknowledge_skip(self) -> Dict[str, Any]:
        return self._post('/api/bilt/chain/acknowledge_skip')

    # ── トレーニング ─────────────────────────────────────────────────────────

    def start_training(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        return self._post('/bilt/train/start', json=cfg, timeout=30)

    def stop_training(self) -> Dict[str, Any]:
        return self._post('/bilt/train/stop')

    def get_training_status(self) -> Dict[str, Any]:
        return self._get('/bilt/train/status')

    def list_project_models(self, project_path: str) -> Dict[str, Any]:
        return self._post('/api/bilt/project/models', json={'project_path': project_path})

    # ── 再ラベリング ──────────────────────────────────────────────────────────

    def get_relabel_models(self, project_path: str) -> Dict[str, Any]:
        return self._post('/bilt/relabel/models', json={'project_path': project_path})

    def start_relabel(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        return self._post('/bilt/relabel/start', json=cfg, timeout=300)

    # ── ワークフロー ──────────────────────────────────────────────────────────

    def get_workflow_status(self) -> Dict[str, Any]:
        return self._get('/api/workflow/status')

    def start_workflow(self, graph: Optional[Dict] = None) -> Dict[str, Any]:
        return self._post('/api/workflow/start', json={'graph': graph} if graph else {})

    def stop_workflow(self) -> Dict[str, Any]:
        return self._post('/api/workflow/stop')

    def resume_workflow(self) -> Dict[str, Any]:
        return self._post('/api/workflow/resume')

    def load_workflow_graph(self, graph: Optional[Dict] = None) -> Dict[str, Any]:
        return self._post('/api/workflow/load_graph', json={'graph': graph})

    def get_saved_workflows(self) -> Dict[str, Any]:
        return self._get('/api/workflows/saved')

    def save_workflow(self, name: str, workflow: Dict[str, Any]) -> Dict[str, Any]:
        return self._post('/api/workflows/save', json={'name': name, 'workflow': workflow})

    def load_workflow_file(self, name: str) -> Dict[str, Any]:
        return self._post('/api/workflows/load_file', json={'name': name})

    def delete_workflow(self, name: str) -> Dict[str, Any]:
        return self._post('/api/workflows/delete', json={'name': name})

    def get_workflow_detections(self) -> Dict[str, Any]:
        return self._get('/api/workflow/detections')

    def get_workflow_streams(self) -> Dict[str, Any]:
        return self._get('/api/workflow/streams')

    # ── テスト推論 ────────────────────────────────────────────────────────────

    def test_image(self, image_bytes: bytes, filename: str = 'image.jpg',
                   model_path: str = '', conf: float = 0.25,
                   iou: float = 0.45) -> Dict[str, Any]:
        """アップロード画像に対して推論を実行し、アノテーション済み画像と検出結果を返す。"""
        url = f'{self.base_url}/bilt/test/image'
        try:
            files = {'image': (filename, image_bytes, 'image/jpeg')}
            data = {'conf': str(conf), 'iou': str(iou)}
            if model_path:
                data['model_path'] = model_path
            # Must NOT send Content-Type: application/json for multipart uploads.
            # Passing None removes the session-level header for this request so
            # requests can auto-set multipart/form-data with the correct boundary.
            resp = self.session.post(url, files=files, data=data, timeout=60,
                                     headers={'Content-Type': None})
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f'BILT test_image error: {e}')
            return {'success': False, 'error': str(e)}


def check_bilt_service(service_url: str = BILT_SERVICE_URL) -> bool:
    try:
        resp = requests.get(f'{service_url}/health', timeout=3)
        return resp.json().get('status') == 'ok'
    except Exception:
        return False
