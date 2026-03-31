# Copyright (C) 2026 Rikiza89
# Licensed under the Apache License, Version 2.0
"""
launcher.py — BILT_Workflow_JA ラベリング & 物体検知ツールデスクトップアプリケーションの起動スクリプト
═══════════════════════════════════════════════════════════════════════════════
pywebviewを使用し (MIT license) Flask web app をネイティブな OS ウィンドウにラップします。

  Windows 10/11 → Microsoft Edge WebView2  (常にインストール on Win 10/11)
  macOS         → WKWebView               (ビルドイン)
  Linux/RPi     → WebKit2GTK              (apt install python3-gi gir1.2-webkit2-4.0)

ライブラリの追加インストール:
    pip install pywebview

起動方法:
    python launcher.py            # ノーマルモード (コンソール出力なし)
    python launcher.py --debug    # デバッグモード (コンソール出力とDevToolsを表示)
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON   = sys.executable
BILT_SERVICE = os.path.join(BASE_DIR, 'bilt_service.py')
APP          = os.path.join(BASE_DIR, 'app.py')
LOGS_DIR     = os.path.join(BASE_DIR, 'logs')

APP_URL          = 'http://127.0.0.1:5000'
BILT_SERVICE_URL = 'http://127.0.0.1:5002'

WINDOW_TITLE  = 'BILT ラベリング & 物体検知'
WINDOW_WIDTH  = 1440
WINDOW_HEIGHT = 900
WINDOW_MIN    = (960, 640)

DEBUG = '--debug' in sys.argv

# ── シングルショット　シャットダウンガード　──────────────────────────────────────
_shutdown_event = threading.Event()


# ── ヘルパー ───────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f'[launcher] {msg}', flush=True)


def _wait_for_server(url: str, timeout: float = 60.0, label: str = '',
                     proc: 'subprocess.Popen | None' = None) -> bool:
    """HTTPレスポンスが返されるまで *url* をポーリングします (4xx/5xx はサーバーが起動していることを意味します)。
    *proc* を渡すとプロセス死活監視を行い、クラッシュ時は即座に False を返します。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        # プロセスがクラッシュした場合は即座に中断
        if proc is not None and proc.poll() is not None:
            _log(f'{label}: プロセスが終了しました (終了コード {proc.returncode})')
            return False
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except urllib.error.HTTPError:
            return True          # サーバーは起動しているが、パスが存在しない
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
    """Launch *script*　を現在使用の Python インタープリタで起動し、出力を *log_path* に保存します."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file = open(log_path, 'w', buffering=1, encoding='utf-8', errors='replace')

    kwargs: dict = dict(
        cwd=BASE_DIR,
        stdout=log_file,
        stderr=log_file,
    )

    if sys.platform == 'win32' and not DEBUG:
        kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW

    proc = subprocess.Popen([PYTHON, script], **kwargs)
    _log(f'{label}起動 (PID {proc.pid}) — ログ: {log_path}')
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
    _log('お疲れ様でした。')


# ── Splash screen ─────────────────────────────────────────────────────────────
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
    font-family: 'Segoe UI', system-ui, sans-serif;
    color: #e6edf3;
    user-select: none;
  }
  .logo { font-size: 3rem; margin-bottom: 1rem; }
  h1 { font-size: 1.6rem; font-weight: 600; margin-bottom: 0.4rem; }
  .sub { font-size: 0.95rem; color: #8b949e; margin-bottom: 2.5rem; }
  .spinner {
    width: 40px; height: 40px;
    border: 3px solid #30363d;
    border-top-color: #58a6ff;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  .status { margin-top: 1.2rem; font-size: 0.85rem; color: #8b949e; min-height: 1.2em; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
  <div class="logo">&#127919;</div>
  <h1>BILT_Workflow_JA ラベリング &amp; 物体検知</h1>
  <p class="sub">サービスを起動中、お待ちください&hellip;</p>
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
    }, 2500);
  </script>
</body>
</html>"""


# ── メイン ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── pywebviewがインストールされているか確認 ──────────────────────────────────────────
    try:
        import webview  # noqa: F401
    except ImportError:
        print(
            '\n[launcher] pywebviewはインストールされていません。\n'
            '  実行:  pip install pywebview\n'
            '  その後再起動:  python launcher.py\n',
            flush=True,
        )
        sys.exit(1)

    import webview

    procs: list = []
    log_files: list = []

    def _shutdown(*args):
        shutdown(procs, log_files, *args)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── 1. スプラッシュ画面を表示してウィンドウを作成 ──────────────────────────────
    window = webview.create_window(
        WINDOW_TITLE,
        html=_SPLASH_HTML,
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        min_size=WINDOW_MIN,
        text_select=True,
        zoomable=True,
    )

    # ── 2. ブートスレッド: bilt_service → app の順に起動 ────────────────────
    def _boot() -> None:
        # ── Step 1: bilt_service を起動 ────────────────────────────────────────
        _log('BILT サービスを起動しています…')
        bilt_svc, bilt_log = _start_process(
            BILT_SERVICE, 'bilt_service',
            os.path.join(LOGS_DIR, 'bilt_service.log')
        )
        procs.append(bilt_svc)
        log_files.append(bilt_log)

        bilt_ready = _wait_for_server(
            BILT_SERVICE_URL + '/health',
            timeout=120,
            label='bilt_service',
            proc=bilt_svc,
        )
        if not bilt_ready:
            _log('警告: BILT サービスが起動しませんでした。BILTトレーニング/検出機能は使用できません。')
        else:
            _log('BILT サービスが起動しました')

        # ── Step 2: Flask アプリを起動 ─────────────────────────────────────────
        _log('Flask アプリを起動しています…')
        flask, app_log = _start_process(APP, 'app',
                                        os.path.join(LOGS_DIR, 'app.log'))
        procs.append(flask)
        log_files.append(app_log)

        app_ready = _wait_for_server(APP_URL, timeout=90, label='app', proc=flask)

        if not app_ready:
            log_hint = os.path.join(LOGS_DIR, 'app.log')
            _log(f'エラー: Flask アプリが起動しません。{log_hint} を確認してください')
            _shutdown()
            window.destroy()
            return

        _log(f'準備完了 — {APP_URL} を開きます')
        window.load_url(APP_URL)

    boot_thread = threading.Thread(target=_boot, daemon=True)

    # ── 3. ウィンドウを開く (閉じられるまでブロック) ─────────────────────────────────
    webview.start(
        func=lambda: boot_thread.start(),
        debug=DEBUG,
        private_mode=False,
    )

    _shutdown()


if __name__ == '__main__':
    main()
