import logging
import os
import time
import threading
from queue import Empty, Queue

import numpy as np

class TrackingWorker(threading.Thread):
    def __init__(self, camera_manager, gaze_estimator, result_queue):
        super().__init__(daemon=True)
        self.camera_manager = camera_manager
        self.gaze_estimator = gaze_estimator
        self.result_queue = result_queue
        self.active = False
        
    def start_worker(self):
        self.active = True
        self.start()
        
    def stop_worker(self):
        self.active = False
        
    def run(self):
        last_ts = 0.0
        while self.active:
            ts = self.camera_manager.get_latest_timestamp()
            if ts > last_ts:
                frame = self.camera_manager.get_frame()
                if frame is not None:
                    result = self.gaze_estimator.process_frame(frame)
                    while not self.result_queue.empty():
                        try:
                            self.result_queue.get_nowait()
                        except Empty:
                            pass
                    self.result_queue.put({"tracking_result": result, "ts": ts})
                last_ts = ts
            else:
                time.sleep(0.005)

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
        if loaded_calibration and "baseline_ear" in loaded_calibration:
            self.gesture_engine.baseline_ear = loaded_calibration["baseline_ear"]
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
        self.tracking_queue = Queue()
        self.tracking_worker = TrackingWorker(self.camera_manager, self.gaze_estimator, self.tracking_queue)
        self.camera_manager.on_disconnect = self._on_camera_disconnect
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
        self.last_tracking_result = None

    def _on_camera_disconnect(self):
        self.action_queue.put({"action": "camera_disconnected"})

    def start(self):
        self.camera_manager.start()
        self.tracking_worker.start_worker()
        self.hotkeys.start()
        self.ui.show()
        self.timer.start(self.config.data["ui"]["processing_interval_ms"])

    def shutdown(self):
        self.timer.stop()
        self.hotkeys.stop()
        self.ui.hide_calibration_overlay()
        self.tracking_worker.stop_worker()
        if self.tracking_worker.is_alive():
            self.tracking_worker.join(1.0)
        self.gaze_estimator.close()
        self.camera_manager.stop()

    def _tick(self):
        self._process_actions()
        now_perf = time.perf_counter()
        
        while not self.tracking_queue.empty():
            try:
                msg = self.tracking_queue.get_nowait()
                self.last_tracking_result = msg["tracking_result"]
            except Empty:
                break

        if not self.camera_manager.is_running() or self.camera_manager.get_frame() is None:
            self._update_calibration_ui_only()
            self._push_runtime_state(face_detected=False, fps=0.0)
            return

        dt = self._update_timing(now_perf)
        tracking_result = self.last_tracking_result
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
        blink_info = self.gesture_engine.detect_blink(tracking_result["blink_landmarks"], tracking_result["head_pose"], now=time.time())
        self.last_blink_info = blink_info
        calibration_event = self._handle_calibration(tracking_result, blink_info)
        dwell_progress = 0.0
        cursor_text = "Cursor: paused"

        if not self.gesture_engine.rest_mode and not self.calibration_manager.active:
            # ANTI-BLINK FREEZE: Prevent cursor jumps during eye blinks
            if blink_info.get("is_blinking", False):
                cursor_text = "المؤشر: متجمد (رمش العين)"
            else:
                # Map raw gaze and tracking data to screen
                screen_pos_raw = self.cursor_mapper.map_gaze_to_screen(tracking_result)
                
                # Apply sensitivity gain centered on the monitor center (optional but common)
                # Or just use the raw mapped position if mapping is 1:1.
                # Here we use the raw mapping as the ground truth.
                screen_pos = screen_pos_raw
                
                smoothed_pos = self.smoothing_filter.filter(np.asarray(screen_pos, dtype=np.float32), dt)
                self.mouse_controller.move_cursor(*smoothed_pos)
                cursor_text = f"المؤشر: {int(smoothed_pos[0])}, {int(smoothed_pos[1])}"

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
            elif action == "camera_disconnected":
                self.camera_manager.stop()
                self.ui.toggle_camera_button.setText("فتح الكاميرا 📹")
                self.last_action = "⚠️ إنقطع اتصال الكاميرا فجأة!"
                self.last_tracking_result = None

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
            
        # IMPORTANT: Calibrate on RAW GAZE to ensure the model captures the full eye range
        # Do not apply sensitivity or scaling here.
        raw_point = np.asarray(tracking_result["normalized_point"], dtype=np.float32)

        event = self.calibration_manager.observe(
            {
                "normalized_point": raw_point,
                "raw_gaze_ratio": tracking_result.get("raw_gaze_ratio"),
                "tracking_quality": tracking_result["tracking_quality"],
                "head_pose": tracking_result["head_pose"],
                "is_blinking": blink_info.get("is_blinking", False),
                "blink_triggered": blink_info["triggered"],
                "raw_ear": blink_info.get("raw_ear", 1.0),
            }
        )
        # ... resto stays the same ...
        overlay_state = event.get("overlay")
        if overlay_state is not None:
            self.ui.show_calibration_overlay(overlay_state)
        
        if event["status"] == "completed":
            calibration_data = event["calibration"]
            self.cursor_mapper.set_calibration(calibration_data)
            if "baseline_ear" in calibration_data:
                self.gesture_engine.baseline_ear = calibration_data["baseline_ear"]
            
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
