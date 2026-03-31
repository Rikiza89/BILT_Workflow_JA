# Copyright (C) 2026 Rikiza89
# Licensed under the Apache License, Version 2.0
"""
launcher_rpi.py — BILT_Workflow_JA スタートアップスクリプト for Raspberry Pi 4
═══════════════════════════════════════════════════════════════════════════════
Raspberry Pi 4 用の起動スクリプトです。
ヘッドレスモードとオプションのGUIモード (pywebview + WebKit2GTK) をサポートします。

動作モード:
  1. ヘッドレス (デフォルト): サービスを起動しブラウザURLを表示します。
     手動でブラウザを開いて http://127.0.0.1:5000 にアクセスします。
  2. GUIモード (--gui オプション): pywebview を使ってネイティブウィンドウを開きます。
     事前に: sudo apt install python3-gi gir1.2-webkit2-4.0 gir1.2-webkit2-4.1

起動方法:
    python launcher_rpi.py            # ヘッドレスモード
    python launcher_rpi.py --gui      # GUIモード (WebKit2GTK が必要)
    python launcher_rpi.py --debug    # ヘッドレス + 詳細ログ出力
    python launcher_rpi.py --gui --debug  # GUI + DevTools

Raspberry Pi 固有の最適化:
  - BILT_FRAME_RATE_LIMIT=15 (デフォルト) でCPU負荷を抑制
  - 初回起動時はモデルダウンロードのため長めのタイムアウトを設定
  - picamera2 は自動検出されます (sudo apt install python3-picamera2)
"""

import os
import sys
import time
import signal
import threading
import subprocess
import urllib.request
import urllib.error

# ── パス ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PYTHON       = sys.executable
BILT_SERVICE = os.path.join(BASE_DIR, 'bilt_service.py')
APP          = os.path.join(BASE_DIR, 'app.py')
LOGS_DIR     = os.path.join(BASE_DIR, 'logs')

APP_URL          = 'http://127.0.0.1:5000'
BILT_SERVICE_URL = 'http://127.0.0.1:5002'

WINDOW_TITLE  = 'BILT - Raspberry Pi 4'
WINDOW_WIDTH  = 1280
WINDOW_HEIGHT = 800
WINDOW_MIN    = (800, 600)

DEBUG    = '--debug' in sys.argv
GUI_MODE = '--gui'   in sys.argv

# ── シングルショット シャットダウンガード ─────────────────────────────────────
_shutdown_event = threading.Event()


# ── ヘルパー ───────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f'[launcher_rpi] {msg}', flush=True)


def _wait_for_server(url: str, timeout: float = 60.0, label: str = '',
                     proc: 'subprocess.Popen | None' = None) -> bool:
    """HTTPレスポンスが返されるまで *url* をポーリングします。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            _log(f'{label}: プロセスが終了しました (終了コード {proc.returncode})')
            return False
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except urllib.error.HTTPError:
            return True
        except urllib.error.URLError as e:
            reason = str(e.reason)
            if 'Connection refused' not in reason and 'timed out' not in reason:
                _log(f'予期しないエラー {label}: {e}')
            time.sleep(0.5)
        except Exception:
            time.sleep(0.5)
    _log(f'TIMEOUT: {label or url} がタイムアウトしました。')
    return False


def _start_process(script: str, label: str, log_path: str):
    """スクリプトを起動し出力を *log_path* に保存します。"""
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file = open(log_path, 'w', buffering=1, encoding='utf-8', errors='replace')
    proc = subprocess.Popen(
        [PYTHON, script],
        cwd=BASE_DIR,
        stdout=log_file if not DEBUG else None,
        stderr=log_file if not DEBUG else None,
    )
    _log(f'{label} 起動 (PID {proc.pid}) — ログ: {log_path}')
    return proc, log_file


def shutdown(procs, log_files, *_) -> None:
    if _shutdown_event.is_set():
        return
    _shutdown_event.set()
    _log('サービスを終了しています…')
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    for p in procs:
        try:
            p.wait(timeout=6)
        except Exception:
            pass
    for f in log_files:
        try:
            f.close()
        except Exception:
            pass
    _log('シャットダウン完了。')


# ── スプラッシュ画面 HTML (GUI モード用) ──────────────────────────────────────
_SPLASH_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #0d1117;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100vh;
    font-family: system-ui, sans-serif;
    color: #e6edf3;
    user-select: none;
  }
  .logo { font-size: 3rem; margin-bottom: 1rem; }
  h1 { font-size: 1.4rem; font-weight: 600; margin-bottom: 0.4rem; }
  .sub { font-size: 0.9rem; color: #8b949e; margin-bottom: 2rem; }
  .rpi { font-size: 0.8rem; color: #58a6ff; margin-bottom: 2.5rem; }
  .spinner {
    width: 36px; height: 36px;
    border: 3px solid #30363d;
    border-top-color: #58a6ff;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  .status { margin-top: 1rem; font-size: 0.82rem; color: #8b949e; min-height: 1.2em; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
  <div class="logo">&#127919;</div>
  <h1>BILT_Workflow_JA ラベリング &amp; 物体検知</h1>
  <p class="sub">サービスを起動中、お待ちください&hellip;</p>
  <p class="rpi">&#127382; Raspberry Pi 4</p>
  <div class="spinner"></div>
  <p class="status" id="s">Initialising&hellip;</p>
  <script>
    const msgs = [
      'BILT サービス起動中\u2026',
      'ウェブインターフェース準備中\u2026',
      '準備完了\u2026',
    ];
    let i = 0;
    setInterval(() => {
      document.getElementById('s').textContent = msgs[Math.min(i++, msgs.length - 1)];
    }, 3000);
  </script>
</body>
</html>"""


# ── ヘッドレスモード ──────────────────────────────────────────────────────────

def run_headless() -> None:
    """サービスを起動してURLをコンソールに表示します。"""
    procs: list = []
    log_files: list = []

    def _shutdown_handler(*args):
        shutdown(procs, log_files, *args)

    signal.signal(signal.SIGINT,  _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    _log('=' * 60)
    _log('BILT — Raspberry Pi 4 ヘッドレスモード')
    _log('=' * 60)

    # Step 1: bilt_service
    _log('BILT サービスを起動しています…')
    bilt_svc, bilt_log = _start_process(
        BILT_SERVICE, 'bilt_service',
        os.path.join(LOGS_DIR, 'bilt_service.log')
    )
    procs.append(bilt_svc)
    log_files.append(bilt_log)
    bilt_ready = _wait_for_server(
        BILT_SERVICE_URL + '/health',
        timeout=180,
        label='bilt_service',
        proc=bilt_svc,
    )
    if not bilt_ready:
        _log('警告: BILT サービスが起動しませんでした。BILT機能は使用できません。')
    else:
        _log('BILT サービスが起動しました')

    # Step 2: Flask app
    _log('Flask アプリを起動しています…')
    flask, app_log = _start_process(APP, 'app',
                                    os.path.join(LOGS_DIR, 'app.log'))
    procs.append(flask)
    log_files.append(app_log)
    app_ready = _wait_for_server(APP_URL, timeout=120, label='app', proc=flask)

    if not app_ready:
        _log(f'エラー: Flask アプリが起動しません。{os.path.join(LOGS_DIR, "app.log")} を確認してください')
        _shutdown_handler()
        return

    _log('')
    _log('=' * 60)
    _log(f'準備完了!  ブラウザで以下のURLを開いてください:')
    _log(f'  {APP_URL}')
    _log('=' * 60)
    _log('停止するには Ctrl+C を押してください。')

    # ── Raspberry Pi でデフォルトブラウザを開こうとします ────────────────────
    try:
        subprocess.Popen(['chromium-browser', '--kiosk', APP_URL],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _log('Chromium を起動しました (キオスクモード)')
    except FileNotFoundError:
        try:
            subprocess.Popen(['chromium', '--kiosk', APP_URL],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            _log('Chromium を起動しました')
        except FileNotFoundError:
            try:
                subprocess.Popen(['xdg-open', APP_URL],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                _log('デフォルトブラウザを起動しました')
            except FileNotFoundError:
                _log('ブラウザを自動起動できませんでした。手動でブラウザを開いてください。')

    # メインスレッドはシャットダウンを待ちます
    try:
        while not _shutdown_event.is_set():
            # プロセスの生存確認
            for p in procs:
                if p.poll() is not None:
                    _log(f'プロセス (PID {p.pid}) が終了しました')
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        _shutdown_handler()


# ── GUI モード (pywebview) ──────────────────────────────────────────────────

def run_gui() -> None:
    """pywebview を使ってネイティブウィンドウで起動します。"""
    try:
        import webview  # noqa: F401
    except ImportError:
        print(
            '\n[launcher_rpi] pywebview がインストールされていません。\n'
            '  実行: pip install pywebview\n'
            '  Linux/RPi 追加パッケージ: sudo apt install python3-gi gir1.2-webkit2-4.0\n'
            '  ヘッドレスモードで再起動します: python launcher_rpi.py\n',
            flush=True,
        )
        run_headless()
        return

    import webview

    procs: list = []
    log_files: list = []

    def _shutdown_handler(*args):
        shutdown(procs, log_files, *args)

    signal.signal(signal.SIGINT,  _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    window = webview.create_window(
        WINDOW_TITLE,
        html=_SPLASH_HTML,
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        min_size=WINDOW_MIN,
        text_select=True,
        zoomable=True,
    )

    def _boot() -> None:
        # Step 1: bilt_service
        _log('BILT サービスを起動しています…')
        bilt_svc, bilt_log = _start_process(
            BILT_SERVICE, 'bilt_service',
            os.path.join(LOGS_DIR, 'bilt_service.log')
        )
        procs.append(bilt_svc)
        log_files.append(bilt_log)
        bilt_ready = _wait_for_server(
            BILT_SERVICE_URL + '/health',
            timeout=180,
            label='bilt_service',
            proc=bilt_svc,
        )
        if not bilt_ready:
            _log('警告: BILT サービスが起動しませんでした。')
        else:
            _log('BILT サービスが起動しました')

        # Step 2: Flask app
        _log('Flask アプリを起動しています…')
        flask, app_log = _start_process(APP, 'app',
                                        os.path.join(LOGS_DIR, 'app.log'))
        procs.append(flask)
        log_files.append(app_log)
        app_ready = _wait_for_server(APP_URL, timeout=120, label='app', proc=flask)

        if not app_ready:
            _log(f'エラー: Flask アプリが起動しません。{os.path.join(LOGS_DIR, "app.log")} を確認してください')
            _shutdown_handler()
            window.destroy()
            return

        _log(f'準備完了 — {APP_URL} を開きます')
        window.load_url(APP_URL)

    boot_thread = threading.Thread(target=_boot, daemon=True)

    webview.start(
        func=lambda: boot_thread.start(),
        debug=DEBUG,
        private_mode=False,
    )

    _shutdown_handler()


# ── エントリーポイント ─────────────────────────────────────────────────────────

def main() -> None:
    _log(f'モード: {"GUI (pywebview)" if GUI_MODE else "ヘッドレス"}')
    _log(f'デバッグ: {"有効" if DEBUG else "無効"}')

    if GUI_MODE:
        run_gui()
    else:
        run_headless()


if __name__ == '__main__':
    main()
