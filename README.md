# BILT_Workflow_JA アノテーション & 検出ツール

**データセット アノテーション**、**BILT モデル トレーニング**、**リアルタイム オブジェクト検出**を 1 つのインターフェイスに組み合わせた統合 Flask Web アプリケーション。
任意のブラウザで実行するか、内蔵ランチャーを介して **スタンドアロン デスクトップ ウィンドウ** (ブラウザは必要ありません) として実行します。
**Windows**、**Linux**、**macOS**、**Raspberry Pi 4** (Pi カメラ モジュールを含む) を完全にサポートします。

---

## 目次

- [機能](#features)
- [アーキテクチャ](#アーキテクチャ)
- [インストール](#installation)
  - [Windows](#windows)
  - [Linux / macOS](#linux--macos)
  - [ラズベリーパイ 4](#raspberry-pi-4)
- [アプリの実行](#running-the-app)
  - [ブラウザモード](#browser-mode)
  - [デスクトップウィンドウモード (ランチャー)](#desktop-window-mode-launcher)
  - [Raspberry Pi 4 ランチャー](#raspberry-pi-4-launcher)
- [利用ガイド](#usage-guide)
  - [アノテーションワークフロー](#annotation-workflow)
  - [検出ワークフロー](#detection-workflow)
  - [ビジュアルワークフローエディタ](#visual-workflow-editor)
    - [ノードタイプ](#node-types)
    - [アラートノード](#alert-node)
    - [アラートペイロード形式](#alert-payload-format)
    - [ローカルホスト アラート ビューア](#localhost-alert-viewer)
  - [チェーン検出設定](#chain-detection-setup)
- [カメラサポート](#camera-support)
- [モデル](#models)
- [プロジェクトフォルダー構造](#プロジェクトフォルダー構造)
- [構成](#configuration)
- [トラブルシューティング](#troubleshooting)

---

## 特徴

### 注釈
- **プロジェクト管理** — 構造化された `train/val` フォルダーを使用してプロジェクトを作成します
- **マルチタスクのラベル付け** — 境界ボックス (`detect`)、ポリゴン (`segment`)、有向ボックス (`OBB`)
- **ライブ カメラ フィード** — WebSocket ストリーム経由で Web カメラから画像を直接キャプチャします

### BILT 検出
- **ゼロショット学習** — BILT (Because I Like Twice) モデルによるトレーニング不要の物体検出
- **リアルタイム BILT 推論** — 独立したバックエンドサービス (port 5002) で動作
- **BILT ワークフローノード** — ビジュアルワークフローエディターで `🔮 BILT検知` ノードを使用可能
- **マルチカメラ BILT** — ワークフロー内の各 BILT 検出ノードに独立したカメラを割り当て可能 (`BiltPerCameraStream`)
- **BILT トレーニング** — ラベル付き画像からゼロから BILT モデルをトレーニング
- **BILT 再ラベリング** — 既存の画像セットに BILT モデルで自動ラベル付け

### デスクトップランチャー
- アプリを **ネイティブ OS ウィンドウ**として開きます — ブラウザーやアドレス バーは使用しません
- **pywebview** (MIT ライセンス) を使用: Windows では Edge WebView2、macOS では WKWebView、Linux/Pi では WebKit2GTK
- バックエンドサービス (BILT / Flask) が自動的に開始され、ウィンドウが閉じられるとシャットダウンされます
- 診断ログは `logs/app.log`、`logs/bilt_service.log` に書き込まれます
- **Raspberry Pi 4 専用ランチャー** (`launcher_rpi.py`) — ヘッドレスモードと GUI モードをサポート

---

## アーキテクチャ

```
app.py            port 5000  統合 Web アプリ — アノテーション、プロキシ API
bilt_service.py   port 5002  BILT 推論バックエンド、WorkflowEngine、トレーニング
bilt_managers.py             共有マネージャー (カメラ、チェーン、画像保存)
bilt_client.py               HTTP クライアント (app → bilt_service)
config.py                    設定
launcher.py                  デスクトップランチャー (Windows / Linux / macOS)
launcher_rpi.py              Raspberry Pi 4 ランチャー (ヘッドレス + GUI)
templates/                   HTML テンプレート
static/                      CSS / JS / アイコン
bilt_models/                 BILT モデル (.pth) + サイドカー JSON ファイル
projects/                    アノテーションプロジェクトデータ
chains/                      保存されたチェーン構成
workflows/                   保存されたワークフローグラフ
datasets/                    キャプチャされた検出データセット
saved_images/                単一検出スナップショット
logs/                        サービスログ (初回実行時に作成)
RASPBERRY_PI_SETUP.md        Raspberry Pi 4 インストールガイド
```

2 つのサービスはローカルホスト経由で通信します。`app.py` はブラウザで開くフロントエンドです。`bilt_service.py` は BILT カメラ・推論・ワークフロー・トレーニングを担当します。

---

## インストール

### ウィンドウ

```bat
:: 1. 仮想環境を作成してアクティブ化する
Python -m venv venv
venv\Scripts\activate

:: 2. bilt ライブラリをローカルインストール
cd bilt
pip install -e .
cd ..

:: 3. (オプション) GPU サポートのために、最初に CUDA を使用して PyTorch をインストールします
:: https://pytorch.org/get-started/locally/

:: 4. 依存関係をインストールする
pip install -r 要件.txt
```

### Linux / macOS

```bash
#1. 仮想環境を作成してアクティブ化する
python3 -m venv venv
ソース venv/bin/activate

# 2. (オプション) GPU サポートのために最初に CUDA を使用して PyTorch をインストールします
# https://pytorch.org/get-started/locally/

# 3. 依存関係をインストールする
pip install -r 要件.txt

# 4. (デスクトップ ランチャーのみ) WebKit2GTK をインストールする — Linux
sudo apt install -y python3-gi gir1.2-webkit2-4.0
```

### Raspberry Pi 4

**Raspberry Pi OS Bookworm (64 ビット)** でテスト済み。詳細な手順は **[RASPBERRY_PI_SETUP.md](RASPBERRY_PI_SETUP.md)** を参照してください。

```bash
# 1. システムパッケージ
sudo apt update
sudo apt install -y python3-pip python3-venv python3-opencv \
                   python3-gi gir1.2-webkit2-4.0

# 2. Pi カメラモジュールのサポート (CSI リボンケーブルカメラを使用する場合)
sudo apt install -y python3-picamera2

# 3. BILT ライブラリをサイドリポジトリとしてクローン (BILT_Workflow_JA の隣に配置)
cd ~
git clone https://github.com/rikiza89/bilt bilt

# 4. 仮想環境を作成してアクティブ化 (--system-site-packages は必須)
python3 -m venv venv --system-site-packages
source venv/bin/activate

# 5. PyTorch CPU ビルドをインストール (ARM64 用)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 6. アプリの依存関係をインストール
pip install -r requirements.txt
```

> **重要:** `--system-site-packages` を使用すると、`apt` でインストールした `python3-opencv` と `python3-picamera2` が venv 内でも使用できます。省略すると `ModuleNotFoundError` が発生します。

---

## アプリの実行

### ブラウザモード

**2 つのターミナル** (または tmux ペイン) を開き、すべて venv をアクティブ化します。

**ターミナル 1 — BILT バックエンド:**
```bash
python bilt_service.py
```

**ターミナル 2 — Web アプリケーション:**
```bash
python app.py
```

次に、ブラウザで **http://127.0.0.1:5000** を開きます。

---

### デスクトップ ウィンドウ モード (ランチャー)

`pywebview` が必要です (すでに `requirements.txt` にあります)。
ランチャーは両方のサービスを自動的に開始します。個別のターミナルは不要です。

```bash
# 通常モード
python launcher.py

# デバッグモード — DevTools を表示し、ターミナルにサービス出力を表示
python launcher.py --debug
```

ウィンドウが開かない場合は `logs/app.log`、`logs/bilt_service.log` でエラーを確認してください。

**プラットフォームに関する注意事項:**

| プラットフォーム | ウェブビューエンジン | 追加インストール |
|---|---|---|
| Windows 10 / 11 | Microsoft Edge WebView2 | 不要 (プリインストール) |
| macOS | WKWebView | 不要 |
| Linux | WebKit2GTK | `sudo apt install python3-gi gir1.2-webkit2-4.0` |

---

### Raspberry Pi 4 ランチャー

Raspberry Pi 4 専用の `launcher_rpi.py` を使用します。

```bash
# ヘッドレスモード (デフォルト) — サービスを起動し、URL を表示して Chromium を自動起動
python launcher_rpi.py

# GUI モード (WebKit2GTK が必要)
python launcher_rpi.py --gui

# デバッグモード
python launcher_rpi.py --debug
```

ヘッドレスモードでは Chromium を自動で開こうとします。起動後は **http://127.0.0.1:5000** に任意のブラウザでアクセスできます。詳細は [RASPBERRY_PI_SETUP.md](RASPBERRY_PI_SETUP.md) を参照してください。

---

## 使用ガイド

### 注釈ワークフロー

1. **Annotation → Projects** に移動し、タスク タイプ (`detect` / `segment` / `OBB`) を選択して新しいプロジェクトを作成します。
2. プロジェクトのワークスペースを開きます。
3. カメラを起動して画像を直接キャプチャするか、ディスクから画像をアップロードします。
4. 各画像にラベルを描画し、保存します。
5. 基本モデルを選択し、**トレーニングの開始** をクリックします。
6. **Colab Training** を使用して、無料のクラウド GPU トレーニング用のトレーニング ノートブックをエクスポートします。

### 検出ワークフロー

1. **検出**に進みます。
2. カメラとモデルを選択し、[**検出の開始**] をクリックします。
3. 必要に応じて、**信頼性**、**IOU**、**最大検出数**を調整します。
4. **カウンター モード** — クラスごとに一意のオブジェクト検出をカウントします。
5. **チェーン モード** — 連続した検出ステップを設定し、**チェーンの開始** をクリックします。
6. [**フレームを保存**] をクリックして、検出された画像をデータセット プロジェクトにキャプチャします。

### マルチカメラのワークフロー

複数のカメラが接続されている場合、ワークフロー キャンバス内の各 **検出ノード** に **カメラ ソース** ドロップダウンが表示されます。複数のアセンブリ ステーションを並行して検査するには、異なるカメラを異なるノードに割り当てます。

```
カメラ 0 ──── 検出ノード A (部品の位置合わせを確認) ─┐
                                                  ├── 終わり
カメラ 1 ──── 検出ノード B (ラベルの有無を確認)   　─┘
```

**仕組み:**
- ワークフロー開始時、すべての **検出ノード** で使用される一意のカメラごとに `PerCameraStream` スレッドが 1 つ起動されます。
- すべての **BILT検出ノード** で使用される一意のカメラごとに `BiltPerCameraStream` スレッドが 1 つ起動されます。
- 各ストリームはカメラとモデルを個別に所有します。共有状態はありません。
- 並列ワークフローブランチ (同じノードから出る 2 つのエッジ) はそれぞれ独自のカメラスレッドを取得するため、両方が真に同時に実行されます。
- **ライブモニター** パネルは、現在実行中のノードのカメラを表示するように自動的に切り替わります。
- ワークフローに含まれていないカメラは影響を受けません。標準の検出ページは引き続き独立して機能します。

**単一カメラのワークフロー**はカメラ割り当てなしでも以前と同じように動作します。

### ビジュアルワークフローエディター

**ワークフロー** タブ (`/workflow`) を使用すると、`WorkflowEngine` が順次 (または並列分岐で) 実行する検出ステップの視覚的なグラフを構築できます。

#### ノードの種類

| ノード | 色 | 目的 |
|------|--------|----------|
| **開始** | 緑 | エントリポイント — すべてのワークフローはここから始まります |
| **BILT検出** | ピンク/フューシャ | カメラで BILT を実行します。BILT モデルを選択して特定クラスを検出するまで待機します |
| **ループ** | 紫 | `body` 分岐を最大 N 回繰り返し、その後 `out` に続きます |
| **アラート** | オレンジ | HTTP POST および/または UDP ブロードキャスト通知を送信します |
| **待機** | 青緑 | 指定した秒数だけ実行を一時停止します |
| **終了** | グレー | ワークフローの実行を終了します |

ノードは **ポート** を介して接続します。すべてのノードには `out` ポートがあります。BILT検出ノードは `success` (しきい値に到達) ポートと `failure` (タイムアウト) ポートも公開します。

**BILT検出ノードのオプション:**

| オプション | デフォルト | 説明 |
|----------|----------|---------------|
| カメラソース | カメラ 0 | このノードに使用する物理カメラ |
| BILT モデル | (選択必須) | `bilt_models/` から `.pth` モデルを選択してロード |
| ターゲットクラス | (すべて) | BILT モデルのクラスから選択 |
| 必要な最低検出数 | 1 | 同時に存在する必要がある一致検出の数 |
| タイムアウト | 30 秒 | ノードが失敗と見なされるまでの待機時間 |
| 失敗時の動作 | アラートして一時停止 | タイムアウト時: アラートをトリガーして一時停止 / 再試行 / スキップ |

---

#### アラート ノード

**アラート ノード**は、ワークフローの実行中に到達すると、グラフの通常のステップとして、または検出ノードがタイムアウトして「障害」ポートに接続されたときに自動的に起動します。

**設定フィールド:**

|フィールド |説明 |
|------|-----------|
|メッセージ |ペイロードに含まれるフリーテキスト文字列 |
| HTTPポスト |アウトバウンド HTTP を有効にします。ターゲット URL を設定する |
| UDPブロードキャスト | UDP を有効にします。 IP (例: `192.168.1.255`) とポート (デフォルトでは `9999`) を設定します。
|スナップショットを含める |カメラフレームのbase64 JPEGをペイロードに添付します。

どちらのチャネルも同じ JSON 本文を送信します。いずれかまたは両方を同時に有効にすることができます。

---

#### アラート ペイロードの形式

すべてのアラートには、HTTP POST 経由で送信されるか UDP 経由で送信されるかにかかわらず、同じ JSON が含まれます。

```json
{
  "message": "あなたの警告メッセージ",
  "タイムスタンプ": "2026-03-18T14:32:45.123456",
  "ソース": "ワークフロー",
  "frame_snapshot": "<base64 JPEG 文字列>"
}
```

`frame_snapshot` は、**スナップショットを含める** がチェックされており、アラートの時点でカメラ フレームが利用可能だった場合にのみ存在します (検出失敗パスのみ。直接アラート ノードは `null` を送信します)。

**HTTP の詳細:** `POST`、`Content-Type: application/json`、5 秒のタイムアウト。
**UDP の詳細:** ブロードキャスト ソケット (`SO_BROADCAST`)、同じ JSON バイト、デフォルト ポート `9999`。

---

#### ローカルホスト アラート ビューア

アプリをローカルで実行する場合、組み込みのアラート レシーバー ページを利用できます。

**ステップ 1 — アラート ノードを構成します:**

HTTP URL を次のように設定します。
```
http://localhost:5000/api/alert_feed
```

**ステップ 2 — ビューアを開きます:**

次の場所に移動します:
```
http://localhost:5000/alert_viewer
```

ビューアは 2 秒ごとに `/api/alert_feed` をポーリングし、以下を表示します。
- メッセージ、タイムスタンプ、ソース、およびフレームがアタッチされたときの「スナップショットを表示」ボタンを含む、スクロール可能な **アラート ログ** (最新が一番上)
- 受信アラートをリアルタイムで収集する右側の**ライブ トースト パネル**
- 接続が生きている間は緑色の点が点滅します。サービスに到達できない場合は灰色に変わります

フィードは、最後の 200 件のアラートをメモリに保存します (`app.py` が再起動されるとリセットされます)。

> **ヒント:** JSON `POST` を受け入れる任意のエンドポイントで HTTP URL を指定するか、任意の UDP クライアントでポート 9999 で UDP パケットをリッスンすることによって、アラートを他のシステムに統合することもできます。

---

### チェーン検出のセットアップ

チェーン ステップは UI で設定するか、`chains/` に保存されている `.json` ファイルを編集して設定します。

```json
{
  "チェーンステップ": [
    { "name": "ステップ 1 – ユーザーが入力します", "classes": { "person": 1 } },
    { "name": "ステップ 2 – アイテムを拾う", "classes": { "bottle": 1 } },
    { "name": "ステップ 3 – 終了", "classes": { "person": 1 } }
  ],
  "チェーンタイムアウト": 10.0,
  "chain_auto_advance": true,
  "チェーン一時停止時間": 5.0
}
```

- `classes` — `"class_name": minimum_count` のマップ。名前はモデルの出力と**正確に**一致する必要があります(大文字と小文字は区別されます)。
- `chain_timeout` — ステップがタイムアウトして次に進む前の秒数。
- `chain_auto_advance` — `true` の場合、ステップが満たされたときにチェーンが自動的に進みます。
- `chain_pause_time` — 完了した完全なシーケンス間で秒単位で一時停止します。

---

## カメラのサポート

|プラットフォーム | USB / 内蔵カメラ | Piカメラモジュール |
|---|---|---|
|ウィンドウズ | DirectShow → MSMF → デフォルト (自動試行) | — |
| Linux / macOS | V4L2 → デフォルト (自動試行) | `picamera2` (インデックス `"picamera2"`) |
|ラズベリーパイ | V4L2 USB カメラ (`/dev/video0` など) | `picamera2` (インデックス `"picamera2"`) |

Linux / Pi では、利用可能なカメラは `/dev/video*` ノードをプローブすることによって検出されます。
`picamera2` がインストールされている場合、**「Raspberry Pi Camera (picamera2)」** エントリがカメラ セレクターに自動的に表示されます。

---

## モデル

### BILT モデル

BILT モデルファイル (`.pth`) を `bilt_models/` ディレクトリに配置します。各モデルにはオプションのサイドカーファイルを追加できます:

| ファイル | 説明 |
|---|---|
| `{stem}.pth` | BILT モデルの重み |
| `{stem}_params.json` | モデルのハイパーパラメータ (任意) |
| `{stem}_rating.json` | モデルの品質スコア (トレーニング後に自動生成) |

BILT モデルは UI の **BILT Detection** タブまたはワークフローの **BILT検出ノード** でロードします。

---

## プロジェクトフォルダー構造

```
projects/
└── my_project/
    ├── train/
    │   ├── images/  ← トレーニング画像
    │   └── labels/  ← YOLO形式ラベルファイル(.txt)
    ├── val/
    │   ├── images/
    │   └── labels/
    ├── classes.txt
    ├── data.yaml
    └── project_config.json ← タスクタイプ（検出/セグメント/obb）
```

---

## 構成

`config.py` の主な設定:

| 設定 | デフォルト | 説明 |
|---|---|---|
| `PORT` | `5000` | Web アプリのポート |
| `BILT_SERVICE_PORT` | `5002` | BILT サービスポート |
| `DEFAULT_CONF_THRESHOLD` | `0.60` | 検出信頼度のしきい値 |
| `DEFAULT_IOU_THRESHOLD` | `0.10` | NMS IOU しきい値 |
| `DEFAULT_CAMERA_WIDTH` | `1280` | カメラキャプチャ幅 |
| `DEFAULT_CAMERA_HEIGHT` | `960` | カメラキャプチャ高さ |
| `DEFAULT_CAMERA_FPS` | `30` | カメラキャプチャ FPS |
| `BILT_FRAME_RATE_LIMIT` | `15` | BILT 最大推論 FPS (CPU 負荷軽減) |
| `FRAME_BUFFER_SIZE` | `1` | OpenCV フレームバッファサイズ |

---

## トラブルシューティング

**「BILT サービス オフライン」バッジが表示される**
→ `bilt_service.py` が起動していることを確認します。`logs/bilt_service.log` でエラーを確認してください。
→ `../bilt/` ディレクトリ (サイドリポジトリ) が存在することを確認します。

**ランチャーがタイムアウト / ウィンドウがすぐに閉じる**
→ `logs/app.log`、`logs/bilt_service.log` でクラッシュの原因を確認してください。
→ ポート 5000、5002 が他のプロセスで使用されていないことを確認してください。

**カメラが見つからない / 起動しない**
→ Linux では `/dev/video0` が存在することを確認: `ls /dev/video*`
→ カメラセレクターで別のインデックス (0、1、2) を試してください。
→ Windows では DirectShow と Media Foundation が自動的に試行されます。

**Pi カメラモジュールがセレクターに表示されない**
→ `picamera2` がインストール済みか確認: `python3 -c "import picamera2; print('ok')"`
→ カメラインターフェイスが有効か確認: `sudo raspi-config` → インターフェイスオプション → カメラ
→ venv が `--system-site-packages` で作成されていることを確認してください。

**BILT ワークフローノードで「No BILT stream」警告が出る**
→ BILT検出ノードのプロパティでモデルを選択し、「↓ 選択したモデルをロード & クラス」をクリックしてください。
→ `bilt_models/` ディレクトリに `.pth` ファイルが存在することを確認してください。

**カメラフィードの色がおかしい**
→ 検出パネルの **RGB バランス** スライダーを使用します。

**トレーニングがすぐに失敗する**
→ `train/images` + `train/labels` にラベル付き画像が存在することを確認してください。
→ `classes.txt` が空でないことを確認してください。

**チェーンステップが進まない**
→ クラス名は**大文字と小文字が区別されます**。ロードされたモデルの出力と正確に一致する必要があります。
→ チェーンステータスパネルのライブ検出リストで検出されているクラスを確認してください。

**Raspberry Pi で `ModuleNotFoundError: No module named 'cv2'`**
→ venv が `--system-site-packages` で作成されていることを確認してください。詳細は [RASPBERRY_PI_SETUP.md](RASPBERRY_PI_SETUP.md) を参照してください。