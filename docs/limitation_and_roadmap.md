# Known Limitations and Roadmap

## Current Limitations
- Requires Intel CPU with iGPU for optimal performance; AMD/ARM support limited.
- Calibration needed per user/session; no automatic drift correction.
- Single face tracking; may switch users in multi-person scenarios.
- No support for rotated monitors beyond basic DPI scaling.
- Rest mode requires manual hotkey; no automatic detection of user absence.

## Roadmap for Improvements
- Add support for AMD GPUs and ARM processors via OpenVINO extensions.
- Implement continuous calibration with drift detection.
- Enhance multi-user detection with face recognition.
- Add support for arbitrary monitor orientations.
- Integrate automatic rest mode based on eye closure duration or face absence.
- Optimize for lower-end hardware with model quantization.
- Add voice commands for accessibility.