# Eye Mouse Control Pro

## Overview
Eye Mouse Control Pro is a Windows desktop application for controlling the mouse using face, eye, and iris tracking from a webcam. The current build uses MediaPipe FaceMesh for refined facial landmarks, a calibration-based screen mapper, persistent settings, runtime diagnostics, and local-only processing.

## Features
- Real-time eye and head tracking with adaptive frame enhancement
- Smart 9-point calibration with stability gating, outlier rejection, and automatic model selection
- Blink clicking, dwell clicking, and pitch-based scrolling
- Global hotkeys for rest mode, click-mode cycling, and calibration
- Multi-monitor targeting with per-monitor calibration awareness
- Live diagnostics for FPS, CPU usage, tracking quality, blink EAR, and head pose
- Saved settings for sensitivity, preview mirroring, click mode, and monitor selection
- Rolling file logs and basic unit tests for critical logic

## Requirements
- Windows 10/11
- Python 3.10
- Webcam capable of 720p/30fps or better

## Installation
From the project root:

```powershell
pip install -r requirements.txt
python src/main.py
```

## First Run
1. Launch the app with `python src/main.py`
2. Choose the target monitor from the control panel
3. Start calibration and follow all 9 points
4. Tune sensitivity and choose `Blink`, `Dwell`, or `Off`
5. Save settings from the side panel

## Hotkeys
- `Ctrl+Alt+E`: toggle rest mode
- `Ctrl+Alt+M`: cycle click mode
- `Ctrl+Alt+C`: start calibration

## Runtime Notes
- The preview can be mirrored without affecting tracking
- Calibration is monitor-specific; recalibrate after switching target monitors
- Logs are written to `logs/eyemouse.log`

## Testing
```powershell
$env:PYTHONPATH='src'
pytest -q
```
