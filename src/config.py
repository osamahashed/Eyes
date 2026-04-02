import json
import os
from copy import deepcopy


DEFAULT_CONFIG = {
    "video": {
        "width": 1280,
        "height": 720,
        "fps": 30,
        "camera_index": 0,
        "backend": "dshow",
        "buffer_size": 1,
    },
    "ui": {
        "mirror_preview": True,
        "show_debug": True,
        "processing_interval_ms": 30,
        "theme": "graphite",
    },
    "tracking": {
        "eye_weight": 1.35,
        "head_weight_x": 0.95,
        "head_weight_y": 0.85,
        "vertical_gain": 1.2,
        "invert_x": True,
        "invert_y": True,
        "screen_margin": 0.08,
        "lost_face_hold_ms": 500,
        "sensitivity": 1.0,
        "face_detection_confidence": 0.5,
        "face_tracking_confidence": 0.5,
    },
    "smoothing": {
        "min_cutoff": 1.0,
        "beta": 0.007,
        "d_cutoff": 1.0,
    },
    "click": {
        "mode": "Blink",
        "ear_threshold": 0.23,
        "min_close_ms": 150,
        "double_blink_window_ms": 300,
        "dwell_ms": 700,
        "dwell_radius_px": 20,
        "scroll_cooldown_ms": 70,
    },
    "scrolling": {
        "pitch_deadzone_deg": 3.0,
        "pitch_gain": 1.0,
        "speed_cap": 10.0,
    },
    "calibration": {
        "rows": 3,
        "cols": 3,
        "margin": 0.12,
        "edge_padding_px": 150,
        "target_radius_px": 18,
        "settle_frames": 10,
        "samples_per_point": 24,
        "stable_window": 8,
        "max_stability": 0.018,
        "min_tracking_quality": 0.45,
        "max_head_yaw": 22.0,
        "max_head_pitch": 18.0,
        "outlier_mad_scale": 2.8,
        "ridge": 0.0005,
        "validation_threshold_px": 140.0,
        "preferred_model": "auto",
        "order": "center-out",
    },
    "hotkeys": {
        "rest_mode": "<ctrl>+<alt>+e",
        "toggle_modes": "<ctrl>+<alt>+m",
        "start_calibration": "<ctrl>+<alt>+c",
    },
    "monitors": {
        "enable_multi_monitor": True,
        "auto_detect_dpi": True,
        "selected_monitor_index": 0,
    },
    "logging": {
        "level": "INFO",
        "enable_performance_logging": True,
        "directory": "logs",
        "filename": "eyemouse.log",
    },
}


class Config:
    def __init__(self, config_path):
        self.config_path = config_path
        self.data = deepcopy(DEFAULT_CONFIG)
        self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_path):
            self.save_config()
            return
        with open(self.config_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        self.data = self._merge_dicts(deepcopy(DEFAULT_CONFIG), loaded)

    def save_config(self):
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set_nested(self, path, value):
        node = self.data
        for segment in path[:-1]:
            if segment not in node or not isinstance(node[segment], dict):
                node[segment] = {}
            node = node[segment]
        node[path[-1]] = value

    def snapshot(self):
        return deepcopy(self.data)

    def _merge_dicts(self, base, override):
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key] = self._merge_dicts(base[key], value)
            else:
                base[key] = value
        return base
