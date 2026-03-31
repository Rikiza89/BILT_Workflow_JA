# Copyright (C) 2026 Rikiza89
# Licensed under the Apache License, Version 2.0
"""
BILT_Workflow_JA ラベリング + 物体検知アプリケーションの共有設定ファイル
このファイルは、アプリケーション全体で使用される設定を定義します。
"""

import os
import sys

if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-change-this-in-production'

    # ウエブアプリ
    HOST = '127.0.0.1'
    PORT = 5000
    DEBUG = True

    # BILT バックエンドサービス
    BILT_SERVICE_HOST = '127.0.0.1'
    BILT_SERVICE_PORT = 5002

    # フォルダー構成
    BASE_DIR = BASE_DIR
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    BILT_MODELS_DIR = os.path.join(BASE_DIR, 'bilt_models')
    SAVED_IMAGES_DIR = os.path.join(BASE_DIR, 'saved_images')
    DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')
    PROJECTS_DIR = os.path.join(BASE_DIR, 'projects')
    CHAINS_DIR = os.path.join(BASE_DIR, 'chains')
    WORKFLOWS_DIR = os.path.join(BASE_DIR, 'workflows')

    # カメラ初期設定
    DEFAULT_CAMERA_WIDTH = 1280
    DEFAULT_CAMERA_HEIGHT = 960
    DEFAULT_CAMERA_FPS = 30
    MAX_CAMERA_INDEX = 3
    CAMERA_FLIP_HORIZONTAL = True   # カメラが左右反転している場合はTrue

    # 物体検知用デフォルト設定
    DEFAULT_CONF_THRESHOLD = 0.60
    DEFAULT_IOU_THRESHOLD = 0.10
    DEFAULT_MAX_DETECTIONS = 10

    # パフォーマンス
    FRAME_RATE_LIMIT = 30
    BILT_FRAME_RATE_LIMIT = 15
    FRAME_BUFFER_SIZE = 1

    # 画像品質
    SAVE_IMAGE_QUALITY = 99
    MAX_IMAGE_SIZE = (3840, 2160)

    # ログ
    LOG_LEVEL = 'INFO'
    LOG_FILE = os.path.join(BASE_DIR, 'app.log')

    MAX_CONTENT_LENGTH = 16 * 1024 * 1024

    # BILT モデル設定
    BILT_SUPPORTED_MODEL_FORMATS = ['.pth']

    @staticmethod
    def create_directories():
        for d in [Config.MODELS_DIR, Config.BILT_MODELS_DIR, Config.SAVED_IMAGES_DIR,
                  Config.DATASETS_DIR, Config.PROJECTS_DIR, Config.CHAINS_DIR, Config.WORKFLOWS_DIR]:
            os.makedirs(d, exist_ok=True)


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig,
}
