import logging
import os
import time
from queue import Empty, Queue

import numpy as np

try:
    import psutil
except Exception:
    psutil = None

from qt_compat import QtCore
from calibration_manager import CalibrationManager
from camera_manager import CameraManager
from cursor_mapper import CursorMapper
from gaze_estimator import GazeEstimator
from gesture_engine import GestureEngine
from hotkey_manager import HotkeyManager
from mouse_controller import MouseController
from smoothing_filter import SmoothingFilter
from ui_controller import UIController


class AppController(QtCore.QObject):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.action_queue = Queue()
        calibration_path = "config/calibration.json"

        self.camera_manager = CameraManager(config.data)
        self.calibration_manager = CalibrationManager(calibration_path, config.data)
        calibration_exists = os.path.exists(calibration_path)
        loaded_calibration = self.calibration_manager.load_calibration()
        self.cursor_mapper = CursorMapper(self.calibration_manager.screen_transform, config.data)
        self.gesture_engine = GestureEngine(config.data)
        self.smoothing_filter = SmoothingFilter(config.data)
        self.mouse_controller = MouseController()
        self.gaze_estimator = GazeEstimator(config.data)
        self.ui = UIController(
            self.camera_manager,
            self.gesture_engine,
            config.data,
            self.cursor_mapper.monitors,
            self.action_queue,
        )
        self.hotkeys = HotkeyManager(config.data, self.action_queue)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._tick)

        self.last_frame_time = None
        self.last_face_seen = time.perf_counter()
        self.last_runtime_push = 0.0
        self.last_cpu_sample = 0.0
        self.cpu_percent = "n/a"
        self.fps_samples = []
        self.last_blink_info = None
        self.last_action = "recalibrate for new direction mapping" if calibration_exists and loaded_calibration is None else "none"

    def start(self):
        self.camera_manager.start()
        self.hotkeys.start()
        self.ui.show()
        self.timer.start(self.config.data["ui"]["processing_interval_ms"])

    def shutdown(self):
        self.timer.stop()
        self.hotkeys.stop()
        self.ui.hide_calibration_overlay()
        self.gaze_estimator.close()
        self.camera_manager.stop()

    def _tick(self):
        self._process_actions()
        frame = self.camera_manager.get_frame()
        now_perf = time.perf_counter()

        if frame is None:
            self._update_calibration_ui_only()
            self._push_runtime_state(face_detected=False, fps=0.0)
            return

        dt = self._update_timing(now_perf)
        tracking_result = self.gaze_estimator.process_frame(frame)
        self.ui.set_tracking_result(tracking_result)

        if tracking_result is None:
            self._update_calibration_ui_only()
            if now_perf - self.last_face_seen > self.config.data["tracking"]["lost_face_hold_ms"] / 1000.0:
                self.smoothing_filter.reset()
                self.gesture_engine.reset_transient_state()
            self._push_runtime_state(face_detected=False, fps=self._current_fps())
            return

        self.last_face_seen = now_perf
        self.cursor_mapper.set_monitor(self.ui.get_selected_monitor_index())

        raw_point = np.asarray(tracking_result["normalized_point"], dtype=np.float32)
        blink_info = self.gesture_engine.detect_blink(tracking_result["blink_landmarks"], now=time.time())
        self.last_blink_info = blink_info
        calibration_event = self._handle_calibration(tracking_result, blink_info)
        dwell_progress = 0.0
        cursor_text = "Cursor: paused"

        if not self.gesture_engine.rest_mode and not self.calibration_manager.active:
            sensitivity = self.ui.get_sensitivity()
            # Apply sensitivity centered at 0.5 (center of viewport)
            adjusted_point = np.clip(0.5 + (raw_point - 0.5) * sensitivity, 0.0, 1.0)
            screen_pos = self.cursor_mapper.map_gaze_to_screen(adjusted_point)
            smoothed_pos = self.smoothing_filter.filter(np.asarray(screen_pos, dtype=np.float32), dt)
            self.mouse_controller.move_cursor(*smoothed_pos)
            cursor_text = f"Cursor: {int(smoothed_pos[0])}, {int(smoothed_pos[1])}"

            click_mode = self.gesture_engine.click_mode
            if click_mode == "Blink" and blink_info["triggered"]:
                self.mouse_controller.click()
                self.last_action = "blink click"
            elif click_mode == "Dwell":
                dwell_info = self.gesture_engine.detect_dwell(np.asarray(smoothed_pos, dtype=np.float32), time.time())
                dwell_progress = dwell_info["progress"]
                if dwell_info["triggered"]:
                    self.mouse_controller.click()
                    self.last_action = "dwell click"
            else:
                self.gesture_engine.reset_transient_state()

            scroll_delta = self.gesture_engine.detect_scroll(tracking_result["head_pose"], now=time.time())
            if scroll_delta:
                self.mouse_controller.scroll(0, scroll_delta)
                self.last_action = f"scroll {scroll_delta}"
        else:
            self.smoothing_filter.reset()

        self._push_runtime_state(
            face_detected=True,
            fps=self._current_fps(),
            tracking_result=tracking_result,
            blink_info=blink_info,
            dwell_progress=dwell_progress,
            cursor_text=cursor_text,
            calibration_event=calibration_event,
        )

    def _update_calibration_ui_only(self):
        """Updates the calibration overlay even if no face is detected."""
        if not self.calibration_manager.active:
            return
        overlay_state = self.calibration_manager.get_overlay_state()
        if overlay_state:
            # Override hint if no face is detected
            overlay_state["hint"] = "⚠️ لم يتم اكتشاف وجه! يرجى التموضع أمام الكاميرا"
            self.ui.show_calibration_overlay(overlay_state)

    def _process_actions(self):
        while True:
            try:
                message = self.action_queue.get_nowait()
            except Empty:
                return
            action = message["action"]
            payload = message.get("payload")
            if action == "toggle_rest_mode":
                enabled = self.gesture_engine.toggle_rest_mode()
                self.ui.set_rest_mode(enabled)
                self.last_action = f"rest mode {'on' if enabled else 'off'}"
            elif action == "set_rest_mode":
                self.gesture_engine.rest_mode = bool(payload)
                self.ui.set_rest_mode(bool(payload))
                self.last_action = f"rest mode {'on' if payload else 'off'}"
            elif action == "cycle_click_mode":
                new_mode = self.gesture_engine.cycle_click_mode()
                self.ui.set_click_mode(new_mode)
                self.last_action = f"click mode {new_mode.lower()}"
            elif action == "set_click_mode":
                new_mode = self.gesture_engine.set_click_mode(payload)
                self.last_action = f"click mode {new_mode.lower()}"
            elif action == "set_monitor":
                self.cursor_mapper.set_monitor(int(payload))
                self.last_action = f"monitor {int(payload) + 1}"
            elif action == "start_calibration":
                self._start_calibration()
            elif action == "reset_calibration":
                self.calibration_manager.clear_calibration()
                self.cursor_mapper.set_calibration(None)
                self._start_calibration()
                self.last_action = "تم إعادة تشغيل المعايرة من جديد"
            elif action == "save_settings":
                self._save_settings()
                self.last_action = "تم حفظ الإعدادات بنجاح"
            elif action == "toggle_camera":
                if self.camera_manager.is_running():
                    self.camera_manager.stop()
                    self.ui.toggle_camera_button.setText("فتح الكاميرا 📹")
                    self.last_action = "تم إيقاف الكاميرا"
                else:
                    self.camera_manager.start()
                    self.ui.toggle_camera_button.setText("إيقاف الكاميرا 📹")
                    self.last_action = "جاري فتح الكاميرا..."

    def _start_calibration(self):
        # Ensure camera is active
        if not self.camera_manager.is_running():
            self.camera_manager.start()
            self.ui.toggle_camera_button.setText("إيقاف الكاميرا 📹")
            
        monitor_index = self.ui.get_selected_monitor_index()
        self.cursor_mapper.set_monitor(monitor_index)
        calibration_monitor = self.ui.get_calibration_monitor_geometry(monitor_index)
        overlay_state = self.calibration_manager.begin(calibration_monitor, monitor_index)
        self.smoothing_filter.reset()
        self.gesture_engine.reset_transient_state()
        self.ui.show_calibration_overlay(overlay_state)
        self.last_action = "بدأت المعايرة"

    def _handle_calibration(self, tracking_result, blink_info):
        if not self.calibration_manager.active:
            return None
            
        # During calibration, we MUST use the same sensitive point if we want the mapping to be identical
        sensitivity = self.ui.get_sensitivity()
        raw_point = np.asarray(tracking_result["normalized_point"], dtype=np.float32)
        adjusted_point = np.clip(0.5 + (raw_point - 0.5) * sensitivity, 0.0, 1.0)

        event = self.calibration_manager.observe(
            {
                "normalized_point": adjusted_point,
                "tracking_quality": tracking_result["tracking_quality"],
                "head_pose": tracking_result["head_pose"],
                "blink_triggered": blink_info["triggered"],
            }
        )
        # ... resto stays the same ...
        overlay_state = event.get("overlay")
        if overlay_state is not None:
            self.ui.show_calibration_overlay(overlay_state)
        
        if event["status"] == "completed":
            calibration_data = event["calibration"]
            self.cursor_mapper.set_calibration(calibration_data)
            
            # Show results in overlay instead of hiding it
            results_overlay = {
                "monitor": self.calibration_manager.monitor,
                "point": None,
                "index": 0,
                "total": 0,
                "results_report": calibration_data.get("accuracy_report"),
                "hint": "تمت المعايرة الأسطورية بنجاح!"
            }
            self.ui.show_calibration_overlay(results_overlay)
            self.last_action = "calibration completed"
        return event

    def _save_settings(self):
        settings = self.ui.snapshot_settings()
        self.config.set_nested(["click", "mode"], settings["click_mode"])
        self.config.set_nested(["tracking", "sensitivity"], settings["sensitivity"])
        self.config.set_nested(["ui", "mirror_preview"], settings["mirror_preview"])
        self.config.set_nested(["ui", "show_debug"], settings["show_debug"])
        self.config.set_nested(["monitors", "selected_monitor_index"], settings["selected_monitor_index"])
        self.config.save_config()
        self.ui.hide_calibration_overlay()

    def _update_timing(self, now_perf):
        if self.last_frame_time is None:
            self.last_frame_time = now_perf
            return 1 / max(self.config.data["video"]["fps"], 1)
        dt = max(now_perf - self.last_frame_time, 1e-3)
        self.last_frame_time = now_perf
        self.fps_samples.append(1.0 / dt)
        self.fps_samples = self.fps_samples[-30:]
        return dt

    def _current_fps(self):
        if not self.fps_samples:
            return 0.0
        return float(sum(self.fps_samples) / len(self.fps_samples))

    def _push_runtime_state(
        self,
        face_detected,
        fps,
        tracking_result=None,
        blink_info=None,
        dwell_progress=0.0,
        cursor_text="Cursor: n/a",
        calibration_event=None,
    ):
        now = time.perf_counter()
        if psutil is not None and now - self.last_cpu_sample > 1.0:
            self.cpu_percent = f"{psutil.cpu_percent(interval=None):.0f}%"
            self.last_cpu_sample = now

        calibration_state = self.cursor_mapper.calibration_status()
        if self.calibration_manager.active:
            overlay_state = self.calibration_manager.get_overlay_state()
            accepted = overlay_state["accepted_samples"] if overlay_state else 0
            target = overlay_state["target_samples"] if overlay_state else 0
            calibration_text = f"Smart calibration: sampling {accepted}/{target}"
        elif calibration_state["active"]:
            error_px = calibration_state["error_px"]
            model_name = calibration_state.get("model_name") or "affine"
            validation = calibration_state.get("validation_state") or "unknown"
            calibration_text = f"Smart calibration: {model_name} ({error_px:.1f}px, {validation})"
        else:
            calibration_text = "Calibration: raw mapping"

        pose_text = "Head pose: yaw 0 | pitch 0 | roll 0"
        tracking_quality = 0.0
        blink_ear = "n/a"
        if tracking_result is not None:
            pose = tracking_result["head_pose"]
            pose_text = f"Head pose: yaw {pose['yaw']:.1f} | pitch {pose['pitch']:.1f} | roll {pose['roll']:.1f}"
            tracking_quality = tracking_result["tracking_quality"]
        if blink_info is not None:
            blink_ear = f"{blink_info['ear']:.3f}"

        runtime_state = {
            "face_detected": face_detected,
            "fps": fps,
            "cpu_percent": self.cpu_percent,
            "calibration_text": calibration_text,
            "cursor_text": cursor_text,
            "pose_text": pose_text,
            "tracking_quality": tracking_quality,
            "blink_ear": blink_ear,
            "dwell_progress": dwell_progress,
            "last_action": self.last_action,
        }

        if calibration_event and calibration_event.get("status") == "collecting":
            overlay_state = calibration_event.get("overlay") or {}
            runtime_state["last_action"] = overlay_state.get("hint", "calibration sampling")

        self.ui.set_runtime_state(runtime_state)
