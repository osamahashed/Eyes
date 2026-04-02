import time

import numpy as np


class GestureEngine:
    CLICK_MODES = ["Blink", "Dwell", "Off"]

    def __init__(self, config):
        self.config = config
        self.ear_history = []
        self.dwell_start = None
        self.dwell_center = None
        self.last_click = 0.0
        self.last_scroll = 0.0
        self.rest_mode = False
        self.blink_start = None
        self.blink_triggered = False
        self.click_mode = config["click"].get("mode", "Blink")

    def set_click_mode(self, click_mode):
        if click_mode in self.CLICK_MODES:
            self.click_mode = click_mode
        return self.click_mode

    def cycle_click_mode(self):
        current_index = self.CLICK_MODES.index(self.click_mode)
        self.click_mode = self.CLICK_MODES[(current_index + 1) % len(self.CLICK_MODES)]
        return self.click_mode

    def detect_blink(self, landmarks, now=None):
        now = now or time.time()
        left_eye = np.asarray(landmarks["left_eye"], dtype=np.float32)
        right_eye = np.asarray(landmarks["right_eye"], dtype=np.float32)
        left_ear = self._calculate_ear(left_eye)
        right_ear = self._calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0

        self.ear_history.append((now, ear))
        self.ear_history = [sample for sample in self.ear_history if now - sample[0] < 1.0]

        threshold = self.config["click"]["ear_threshold"]
        min_close_s = self.config["click"]["min_close_ms"] / 1000.0
        cooldown_s = max(self.config["click"]["double_blink_window_ms"] / 1000.0, 0.2)

        triggered = False
        if ear < threshold:
            if self.blink_start is None:
                self.blink_start = now
            elif not self.blink_triggered and now - self.blink_start >= min_close_s:
                if now - self.last_click >= cooldown_s:
                    self.last_click = now
                    self.blink_triggered = True
                    triggered = True
        else:
            self.blink_start = None
            self.blink_triggered = False

        return {
            "triggered": triggered,
            "ear": ear,
            "left_ear": left_ear,
            "right_ear": right_ear,
        }

    def detect_dwell(self, cursor_pos, current_time):
        cursor_pos = np.asarray(cursor_pos, dtype=np.float32)
        if self.dwell_start is None or self.dwell_center is None:
            self.dwell_start = current_time
            self.dwell_center = cursor_pos
            return {"triggered": False, "progress": 0.0}

        radius = float(self.config["click"]["dwell_radius_px"])
        if np.linalg.norm(cursor_pos - self.dwell_center) > radius:
            self.dwell_start = current_time
            self.dwell_center = cursor_pos
            return {"triggered": False, "progress": 0.0}

        dwell_s = self.config["click"]["dwell_ms"] / 1000.0
        progress = min((current_time - self.dwell_start) / max(dwell_s, 1e-6), 1.0)
        triggered = progress >= 1.0
        if triggered:
            self.dwell_start = current_time
        return {"triggered": triggered, "progress": progress}

    def detect_scroll(self, head_pose, now=None):
        now = now or time.time()
        cooldown = self.config["click"].get("scroll_cooldown_ms", 70) / 1000.0
        if now - self.last_scroll < cooldown:
            return 0
        pitch = float(head_pose.get("pitch", 0.0))
        deadzone = float(self.config["scrolling"]["pitch_deadzone_deg"])
        if abs(pitch) <= deadzone:
            return 0
        self.last_scroll = now
        direction = 1 if pitch > 0 else -1
        speed = min(abs(pitch) * self.config["scrolling"]["pitch_gain"], self.config["scrolling"]["speed_cap"])
        return int(direction * speed)

    def toggle_rest_mode(self):
        self.rest_mode = not self.rest_mode
        return self.rest_mode

    def reset_transient_state(self):
        self.dwell_start = None
        self.dwell_center = None
        self.blink_start = None
        self.blink_triggered = False

    def _calculate_ear(self, eye_landmarks):
        if len(eye_landmarks) < 6:
            return 1.0
        a = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        b = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        c = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        if c < 1e-6:
            return 1.0
        return float((a + b) / (2.0 * c))
