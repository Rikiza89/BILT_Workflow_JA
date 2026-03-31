# BILT_Workflow_JA on Raspberry Pi 4 — Setup Guide

## Requirements

| Item | Minimum |
|------|---------|
| Hardware | Raspberry Pi 4 Model B (4 GB RAM recommended, 2 GB works) |
| OS | Raspberry Pi OS Bookworm (64-bit, Desktop or Lite) |
| Python | 3.11+ (included in Pi OS Bookworm) |
| Storage | 8 GB free (models + venv) |
| Camera | USB camera **or** Raspberry Pi Camera Module v3 (picamera2) |

---

## 1. System packages

```bash
sudo apt update && sudo apt upgrade -y

# OpenCV system dependencies
sudo apt install -y \
    python3-dev python3-pip python3-venv \
    libopencv-dev python3-opencv \
    libatlas-base-dev libjasper-dev \
    libhdf5-dev libhdf5-serial-dev \
    libqt5gui5 libqt5webkit5 libqt5test5 \
    libjpeg-dev libpng-dev libtiff-dev \
    libv4l-dev v4l-utils

# Raspberry Pi Camera Module v3 (skip if using USB camera only)
sudo apt install -y python3-picamera2

# Optional: launcher GUI mode (WebKit2GTK)
sudo apt install -y python3-gi gir1.2-webkit2-4.0 gir1.2-webkit2-4.1
```

> **Note:** `python3-opencv` and `python3-picamera2` are installed as **system packages**.
> Your virtual environment **must** use `--system-site-packages` so these are visible inside it.

---

## 2. Clone repositories

Both `BILT_Workflow_JA` and `bilt` must be cloned as **sibling directories**:

```bash
cd ~
git clone https://github.com/rikiza89/BILT_Workflow_JA BILT_Workflow_JA
git clone https://github.com/rikiza89/bilt BILT_Workflow_JA
```

Expected layout after cloning:
```
~/
├── BILT_Workflow_JA/    ← this repository
└── bilt/         ← BILT library (sibling, not inside BILT_Workflow_JA)
```

The app discovers the BILT library automatically via relative path (`../bilt/`).
You do **not** need to `pip install bilt`.

---

## 3. Create virtual environment

> **Critical:** use `--system-site-packages` so that `python3-opencv`, `python3-picamera2`,
> and other system-level binary packages installed in step 1 are available inside the venv.

```bash
cd ~/BILT_Workflow_JA
python3 -m venv venv --system-site-packages
source venv/bin/activate
```

Verify the venv is active (prompt shows `(venv)`):
```bash
which python   # should print ~/BILT_Workflow_JA/venv/bin/python
```

---

## 4. Install Python dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# PyTorch (CPU-only build — optimised for ARM64 / Raspberry Pi)
# Use the official PyTorch ARM wheel or the pip index:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# All remaining dependencies
pip install -r requirements.txt
```

> **`opencv-python` vs system OpenCV:**
> The `requirements.txt` lists `opencv-python>=4.8`.
> On Raspberry Pi, `python3-opencv` from `apt` is usually sufficient and faster to install.
> If `pip install opencv-python` fails due to a build error, skip it — the system package
> (already accessible via `--system-site-packages`) will be used automatically.

### Verify key packages

```bash
python -c "import cv2; print('OpenCV', cv2.__version__)"
python -c "import torch; print('PyTorch', torch.__version__)"
python -c "from PIL import Image; print('Pillow OK')"
python -c "import flask; print('Flask OK')"
```

---

## 5. Install BILT library

BILT is loaded directly from the sibling `~/bilt/` directory — **no pip install needed**.
However, you must install its Python dependencies:

```bash
# From inside your activated venv:
pip install -r ~/bilt/requirements.txt
```

If `~/bilt/requirements.txt` does not exist, install manually:

```bash
pip install torch torchvision Pillow numpy
```

Verify BILT is accessible:

```bash
python -c "
import sys
sys.path.insert(0, '../bilt')
from bilt import BILT
print('BILT OK')
"
```

---

## 6. Camera setup

### USB camera (recommended for beginners)

Plug in a USB webcam and verify it is detected:

```bash
v4l2-ctl --list-devices
ls /dev/video*
```

No additional configuration needed — the app auto-detects USB cameras.

### Raspberry Pi Camera Module v3 (picamera2)

1. Enable the camera interface:
   ```bash
   sudo raspi-config
   # → Interface Options → Camera → Enable
   ```

2. Reboot:
   ```bash
   sudo reboot
   ```

3. Test picamera2:
   ```bash
   python3 -c "import picamera2; print('picamera2 OK')"
   ```

4. In the BILT workflow page, select **"Raspberry Pi Camera (picamera2)"** from the camera dropdown in a BILT detection node.

---

## 7. Launch BILT_Workflow_JA

### Headless mode (recommended for Pi without display)

```bash
cd ~/BILT_Workflow_JA
source venv/bin/activate
python launcher_rpi.py
```

The launcher will:
1. Start `bilt_service.py` (port 5002)
2. Start `app.py` (port 5000)
3. Try to open Chromium automatically
4. Print `http://127.0.0.1:5000` — open this URL in any browser

To stop all services: **Ctrl+C**

### GUI mode (requires display + WebKit2GTK)

```bash
python launcher_rpi.py --gui
```

### Debug mode (verbose output)

```bash
python launcher_rpi.py --debug
```

### Manual launch (without launcher)

```bash
cd ~/BILT_Workflow_JA
source venv/bin/activate

# Terminal 1
python bilt_service.py

# Terminal 2
python app.py
```

---

## 8. Service log files

| Log file | Service |
|----------|---------|
| `logs/bilt_service.log` | BILT detection + workflow engine |
| `logs/app.log` | Flask web UI |
| `bilt_service.log` | (root, when not using launcher) |

---

## 9. Autostart on boot (optional)

Create a systemd service to start BILT automatically:

```bash
sudo nano /etc/systemd/system/bilt.service
```

Paste the following (adjust paths if you cloned to a different location):

```ini
[Unit]
Description=BILT Detection Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/BILT_Workflow_JA
ExecStart=/home/pi/BILT_Workflow_JA/venv/bin/python /home/pi/BILT_Workflow_JA/launcher_rpi.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable bilt
sudo systemctl start bilt
sudo systemctl status bilt
```

---

## 10. Performance tips for Raspberry Pi 4

| Setting | Recommendation |
|---------|---------------|
| BILT frame rate | Default 15 FPS — already optimised for RPi |
| Resolution | 640×480 or 1280×720 for detection; avoid 1920×1080 |
| Swap | `sudo dphys-swapfile swapoff && sudo nano /etc/dphys-swapfile` → `CONF_SWAPSIZE=2048` → `sudo dphys-swapfile setup && sudo dphys-swapfile swapon` |
| Cooling | Use a heatsink + fan; CPU throttles at 80°C |
| Overclocking | Pi OS `raspi-config` → Performance → Overclock (optional) |

---

## 11. Common issues

### `ModuleNotFoundError: No module named 'cv2'`
```bash
# Check that --system-site-packages was used:
python -c "import sys; print([p for p in sys.path if 'dist-packages' in p])"
# Should list /usr/lib/python3/dist-packages

# If empty, recreate the venv:
deactivate
rm -rf venv
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install -r requirements.txt
```

### `ModuleNotFoundError: No module named 'picamera2'`
```bash
sudo apt install -y python3-picamera2
# Then recreate venv with --system-site-packages (see above)
```

### `ModuleNotFoundError: No module named 'bilt'`
```bash
# Verify sibling directory exists:
ls ~/bilt/

# Verify sys.path in bilt_service.py adds ../bilt:
# The service auto-adds ~/bilt/ to sys.path if it exists.
# If you cloned bilt elsewhere, set BILT_REPO_DIR in bilt_service.py.
```

### Port already in use
```bash
sudo lsof -i :5000
sudo kill -9 <PID>
# Same for port 5002
```

### Low FPS / high CPU temperature
- Reduce camera resolution in the BILT detection node settings
- Ensure adequate cooling (passive heatsink minimum)

---

## 12. Tested configuration

| Component | Version |
|-----------|---------|
| Raspberry Pi OS | Bookworm 64-bit (2024-11+) |
| Python | 3.11 |
| PyTorch | 2.x (CPU) |
| OpenCV | 4.8+ |
| picamera2 | 0.3+ |
| Flask | 2.3+ |
