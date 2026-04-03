from queue import Queue
from types import SimpleNamespace

import cv2
import numpy as np

from qt_compat import QtWidgets, QtGui, QtCore, Qt

QMainWindow = QtWidgets.QMainWindow
QLabel = QtWidgets.QLabel
QVBoxLayout = QtWidgets.QVBoxLayout
QHBoxLayout = QtWidgets.QHBoxLayout
QGridLayout = QtWidgets.QGridLayout
QWidget = QtWidgets.QWidget
QSlider = QtWidgets.QSlider
QCheckBox = QtWidgets.QCheckBox
QComboBox = QtWidgets.QComboBox
QPushButton = QtWidgets.QPushButton
QProgressBar = QtWidgets.QProgressBar
QFrame = QtWidgets.QFrame
QGroupBox = QtWidgets.QGroupBox
QTimer = QtCore.QTimer
QImage = QtGui.QImage
QPixmap = QtGui.QPixmap
QPainter = QtGui.QPainter
QPen = QtGui.QPen
QColor = QtGui.QColor
QBrush = QtGui.QBrush
QFont = QtGui.QFont


class CalibrationOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.config = {}
        self.monitor = None
        self.point = None
        self.index = 0
        self.total = 0
        self.progress = 0.0
        self.status_text = ""
        self.hint_text = ""
        self.quality = 0.0
        self.stability = None
        self.accepted_samples = 0
        self.target_samples = 0
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        
        self.animation_timer = QtCore.QTimer(self)
        self.animation_timer.timeout.connect(self._animate_step)
        self.pulse_phase = 0.0
        self.rotation_angle = 0.0

        # Legendary colors (Neon Cyan / Emerald / Success)
        self.color_primary = QColor(0, 229, 255) # Cyan
        self.color_secondary = QColor(0, 255, 170) # Neon Green / Success
        self.color_warning = QColor(255, 50, 80) # Modern Red
        self.color_bg = QColor(5, 8, 14, 235) # Near-black futuristic
        
        self.capture_flash = 0.0
        self.results_data = None

    def set_config(self, config):
        self.config = config

    def _animate_step(self):
        self.pulse_phase += 0.15
        self.rotation_angle = (self.rotation_angle + 2.0) % 360
        if self.capture_flash > 0:
            self.capture_flash = max(0, self.capture_flash - 0.08)
        self.update()

    def show_target(self, overlay_state):
        prev_index = self.index
        self.monitor = overlay_state["monitor"]
        self.point = overlay_state["point"]
        self.index = overlay_state["index"]
        self.total = overlay_state["total"]
        
        # Trigger flash if we moved to next point
        if self.index > prev_index and prev_index != 0:
            self.capture_flash = 1.0
            
        self.progress = overlay_state.get("progress", 0.0)
        self.accepted_samples = overlay_state.get("accepted_samples", 0)
        self.target_samples = overlay_state.get("target_samples", 0)
        self.quality = overlay_state.get("tracking_quality", 0.0)
        self.stability = overlay_state.get("stability")
        self.hint_text = overlay_state.get("hint", "")
        self.results_data = overlay_state.get("results_report") # Post-calibration
        
        self.status_text = f"LEGENDARY CALIBRATION [{self.index}/{self.total}]"
        if self.results_data:
            self.status_text = "CALIBRATION ANALYSIS COMPLETE"
            
        self.setGeometry(self.monitor["x"], self.monitor["y"], self.monitor["width"], self.monitor["height"])
        
        if not self.animation_timer.isActive():
            self.animation_timer.start(25) # ~40 FPS for smoothness
            
        self.show()
        self.raise_()
        self.update()

    def hide_overlay(self):
        self.animation_timer.stop()
        self.hide()

    def paintEvent(self, _event):
        if self.monitor is None:
            return

        painter = QPainter(self)
        painter.fillRect(self.rect(), self.color_bg)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.results_data:
            self._draw_results_screen(painter)
            painter.end()
            return

        if self.point is None:
            painter.end()
            return

        target_x = int(self.point[0] - self.monitor["x"])
        target_y = int(self.point[1] - self.monitor["y"])
        panel_rect = self._panel_rect(target_x, target_y)
        target_radius = int(self.config.get("target_radius_px", 20))

        # Capture Flash Effect (Legendary)
        if self.capture_flash > 0:
            flash_radius = int(target_radius * (1.0 + 10.0 * (1.0 - self.capture_flash)))
            flash_color = QColor(self.color_secondary)
            flash_color.setAlpha(int(180 * self.capture_flash))
            painter.setPen(QPen(flash_color, 4))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(target_x - flash_radius, target_y - flash_radius, flash_radius * 2, flash_radius * 2)

        pulse_scale = 1.0 + 0.12 * np.sin(self.pulse_phase)
        current_radius = int(target_radius * pulse_scale)

        # Dynamic State color
        current_color = self.color_primary
        if self.stability is not None:
             if self.stability < 0.012: current_color = self.color_secondary
             elif self.stability > 0.025: current_color = self.color_warning

        # Draw glowing background
        glow_rad = int(current_radius * (3.5 + 0.5 * np.sin(self.pulse_phase * 0.5)))
        gradient = QtGui.QRadialGradient(target_x, target_y, glow_rad)
        gradient.setColorAt(0, QColor(current_color.red(), current_color.green(), current_color.blue(), 90))
        gradient.setColorAt(1, QColor(current_color.red(), current_color.green(), current_color.blue(), 0))
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(target_x - glow_rad, target_y - glow_rad, glow_rad * 2, glow_rad * 2)

        # Main center point
        painter.setBrush(QBrush(current_color))
        painter.drawEllipse(target_x - int(current_radius * 0.4), target_y - int(current_radius * 0.4), int(current_radius * 0.8), int(current_radius * 0.8))

        # Rotating outer rim
        painter.translate(target_x, target_y)
        painter.rotate(self.rotation_angle)
        
        rim_pen = QPen(current_color, max(2, int(current_radius*0.12)))
        painter.setPen(rim_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        rect = QtCore.QRectF(-current_radius, -current_radius, current_radius * 2, current_radius * 2)
        span = 70 * 16 
        gap = 20 * 16 
        for i in range(4):
            painter.drawArc(rect, int(i * (span + gap)), int(span))
            
        painter.rotate(-self.rotation_angle) # un-rotate for progress
        
        # Progress Arc (Legendary)
        if self.progress > 0:
            prog_radius = current_radius + 14
            prog_rect = QtCore.QRectF(-prog_radius, -prog_radius, prog_radius * 2, prog_radius * 2)
            prog_pen = QPen(self.color_secondary, 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(prog_pen)
            start_angle = 90 * 16
            span_angle = -int(360 * self.progress * 16)
            painter.drawArc(prog_rect, start_angle, span_angle)

        painter.translate(-target_x, -target_y)

        # --- Panel ---
        painter.setPen(Qt.PenStyle.NoPen)
        panel_bg = QColor(10, 16, 25, 210)
        painter.setBrush(QBrush(panel_bg))
        painter.drawRoundedRect(panel_rect, 18, 18)
        
        painter.setPen(QPen(QColor(current_color.red(), current_color.green(), current_color.blue(), 120), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(panel_rect, 18, 18)

        # Panel Text
        painter.setPen(QPen(current_color, 1))
        title_font = QFont("Segoe UI", 13, QFont.Weight.Bold)
        title_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.2)
        painter.setFont(title_font)
        
        title_flags = int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap)
        painter.drawText(panel_rect.adjusted(24, 22, -24, -20), title_flags, self.status_text)
        
        painter.setPen(QPen(QColor(240, 248, 255), 1))
        painter.setFont(QFont("Segoe UI", 11))
        body_flags = int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap)
        painter.drawText(panel_rect.adjusted(24, 62, -24, -20), body_flags, self.hint_text)
        
        # Bottom Metrics Bar
        y_metrics = panel_rect.bottom() - 40
        painter.setFont(QFont("Consolas", 10))
        painter.setPen(QPen(QColor(140, 160, 180), 1))
        painter.drawText(panel_rect.left() + 24, y_metrics, f"SAMPLES: {self.accepted_samples}/{self.target_samples}")
        
        painter.drawText(panel_rect.left() + 160, y_metrics, f"QA: {int(self.quality*100)}%")
        
        stab_text = "OK" if self.stability is not None and self.stability < 0.015 else "LOCKING..."
        painter.drawText(panel_rect.left() + 250, y_metrics, f"STB: {stab_text}")
        
        painter.end()

    def _draw_results_screen(self, painter):
        w, h = self.width(), self.height()
        center_x, center_y = w // 2, h // 2
        card_w, card_h = 600, 450
        card_rect = QtCore.QRect(center_x - card_w//2, center_y - card_h//2, card_w, card_h)
        
        # Backdrop
        painter.setBrush(QBrush(QColor(10, 20, 30, 245)))
        painter.setPen(QPen(self.color_primary, 2))
        painter.drawRoundedRect(card_rect, 24, 24)
        
        # Title
        painter.setPen(QPen(self.color_secondary, 1))
        painter.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        painter.drawText(card_rect.adjusted(0, 40, 0, 0), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, "Legendary Precision Locked")
        
        # Score
        score = self.results_data.get("accuracy_score_percent", 0.0)
        painter.setFont(QFont("Oswald", 60, QFont.Weight.Bold))
        painter.setPen(QPen(self.color_primary, 1))
        painter.drawText(card_rect.adjusted(0, 110, 0, 0), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, f"{score:.1f}%")
        
        painter.setFont(QFont("Segoe UI", 12))
        painter.setPen(QPen(QColor(200, 220, 240), 1))
        painter.drawText(card_rect.adjusted(0, 210, 0, 0), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, "OVERALL SYSTEM ACCURACY SCORE")
        
        # Metrics details
        painter.setFont(QFont("Consolas", 12))
        metrics_y = card_rect.top() + 260
        error_px = self.results_data.get("mean_error_px", 0.0)
        cv_error = self.results_data.get("cross_validation_error_px", 0.0)
        
        painter.drawText(card_rect.left() + 100, metrics_y, f"> Mean Error:    {error_px:.2f} px")
        painter.drawText(card_rect.left() + 100, metrics_y + 30, f"> System Drift:  {cv_error:.2f} px")
        painter.drawText(card_rect.left() + 100, metrics_y + 60, f"> Points Tested: {len(self.results_data.get('points', []))}")
        
        # Button hint
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        painter.setPen(QPen(self.color_secondary, 1))
        painter.drawText(card_rect.adjusted(0, 0, 0, -40), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom, "Press the 'Save Settings' button to apply this mastery")

    def _panel_rect(self, target_x, target_y):
        panel_width = 340
        panel_height = 200
        padding = 40

        left = padding if target_x > self.width() * 0.52 else self.width() - panel_width - padding
        top = padding if target_y > self.height() * 0.52 else self.height() - panel_height - padding

        left = max(padding, min(left, self.width() - panel_width - padding))
        top = max(padding, min(top, self.height() - panel_height - padding))
        return QtCore.QRect(int(left), int(top), int(panel_width), int(panel_height))


class UIController(QMainWindow):
    def __init__(self, camera_manager, gesture_engine, config, monitors, action_queue=None):
        super().__init__()
        self.camera_manager = camera_manager
        self.gesture_engine = gesture_engine
        self.config = config
        self.monitors = monitors
        self.action_queue = action_queue or Queue()
        self.tracking_result = None
        self.runtime_state = {}
        self.calibration_overlay = CalibrationOverlay()
        self.calibration_overlay.set_config(self.config["calibration"])
        self._init_ui()
        self.preview_timer = QTimer()
        self.preview_timer.timeout.connect(self.update_preview)
        self.preview_timer.start(33)

    def _init_ui(self):
        self.setWindowTitle("Eye Mouse Control Pro | التحكم بالعين أسطوري")
        self.setGeometry(80, 60, 1340, 840)
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background-color: #0b0f1a; color: #e2e8f0; font-family: 'Segoe UI', Arial, sans-serif; }
            QGroupBox { border: 2px solid #243041; border-radius: 12px; margin-top: 20px; padding-top: 20px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 0 8px; color: #00e5ff; }
            QPushButton { background-color: #1a237e; color: white; border: 1px solid #303f9f; border-radius: 10px; padding: 12px 18px; font-weight: 700; font-size: 14px; }
            QPushButton:hover { background-color: #283593; border: 1px solid #00e5ff; }
            QPushButton#action_btn { background-color: #1e40af; }
            QPushButton#danger_btn { background-color: #991b1b; }
            QComboBox, QSlider, QCheckBox, QLabel { font-size: 13px; }
            QComboBox { background-color: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 4px; }
            QProgressBar { border: 1px solid #334155; border-radius: 8px; text-align: center; background-color: #0f172a; height: 18px; }
            QProgressBar::chunk { background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #f97316, stop:1 #fb923c); border-radius: 7px; }
            """
        )

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(18)

        preview_panel = QVBoxLayout()
        preview_panel.setSpacing(12)
        root_layout.addLayout(preview_panel, 3)

        self.video_label = QLabel()
        self.video_label.setMinimumSize(920, 620)
        self.video_label.setStyleSheet("background-color: #020617; border-radius: 12px; border: 1px solid #243041;")
        preview_panel.addWidget(self.video_label)

        summary_box = QGroupBox("حالة النظام الحية (Runtime)")
        summary_layout = QGridLayout(summary_box)
        self.status_label = QLabel("جاري البحث عن وجه...")
        self.metrics_label = QLabel("FPS: 0 | CPU: n/a")
        self.calibration_label = QLabel("المعايرة: قيد الانتظار")
        self.cursor_label = QLabel("موقع الماوس: n/a")
        self.hotkeys_label = QLabel(
            "اختصارات: وضع الراحة Ctrl+Alt+E | تبديل النقر Ctrl+Alt+M | معايرة Ctrl+Alt+C"
        )
        summary_layout.addWidget(self.status_label, 0, 0)
        summary_layout.addWidget(self.metrics_label, 0, 1)
        summary_layout.addWidget(self.calibration_label, 1, 0)
        summary_layout.addWidget(self.cursor_label, 1, 1)
        summary_layout.addWidget(self.hotkeys_label, 2, 0, 1, 2)
        preview_panel.addWidget(summary_box)

        side_panel = QVBoxLayout()
        side_panel.setSpacing(12)
        root_layout.addLayout(side_panel, 2)

        control_box = QGroupBox("لوحة التحكم ⚙️")
        control_layout = QVBoxLayout(control_box)
        control_layout.setSpacing(10)

        self.start_calibration_button = QPushButton("بدء المعايرة الأسطورية 🎯")
        self.start_calibration_button.setObjectName("action_btn")
        self.start_calibration_button.clicked.connect(lambda: self.enqueue_action("start_calibration"))
        control_layout.addWidget(self.start_calibration_button)

        self.toggle_camera_button = QPushButton("إيقاف الكاميرا 📹")
        self.toggle_camera_button.clicked.connect(lambda: self.enqueue_action("toggle_camera"))
        control_layout.addWidget(self.toggle_camera_button)

        self.reset_calibration_button = QPushButton("إعادة ضبط المعايرة 🔄")
        self.reset_calibration_button.setObjectName("danger_btn")
        self.reset_calibration_button.clicked.connect(lambda: self.enqueue_action("reset_calibration"))
        control_layout.addWidget(self.reset_calibration_button)

        self.save_settings_button = QPushButton("حفظ كافة الإعدادات ✅")
        self.save_settings_button.clicked.connect(lambda: self.enqueue_action("save_settings"))
        control_layout.addWidget(self.save_settings_button)

        control_layout.addWidget(QLabel("نمط النقر بالعين"))
        self.click_mode_combo = QComboBox()
        self.click_mode_combo.addItems(["رمشة (Blink)", "توقف (Dwell)", "إيقاف (Off)"])
        # Map values properly
        click_mode_map = {"Blink": 0, "Dwell": 1, "Off": 2}
        self.click_mode_combo.setCurrentIndex(click_mode_map.get(self.config["click"]["mode"], 0))
        self.click_mode_combo.currentTextChanged.connect(self._click_mode_changed)
        control_layout.addWidget(self.click_mode_combo)

        control_layout.addWidget(QLabel("شاشة العرض المراد التحكم بها"))
        self.monitor_combo = QComboBox()
        for idx, monitor in enumerate(self.monitors):
            self.monitor_combo.addItem(f"شاشة رقم {idx + 1} | {monitor.width}x{monitor.height}", idx)
        selected_index = min(self.config["monitors"]["selected_monitor_index"], len(self.monitors) - 1)
        self.monitor_combo.setCurrentIndex(selected_index)
        self.monitor_combo.currentIndexChanged.connect(self._monitor_changed)
        control_layout.addWidget(self.monitor_combo)

        control_layout.addWidget(QLabel("حساسية حركة المؤشر"))
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setRange(1, 15)
        self.sensitivity_slider.setValue(self._sensitivity_to_slider(self.config["tracking"]["sensitivity"]))
        control_layout.addWidget(self.sensitivity_slider)

        self.rest_mode_checkbox = QCheckBox("وضع الراحة (Pause)")
        self.rest_mode_checkbox.stateChanged.connect(self._rest_mode_changed)
        control_layout.addWidget(self.rest_mode_checkbox)

        self.mirror_checkbox = QCheckBox("عكس معاينة الكاميرا (Mirror)")
        self.mirror_checkbox.setChecked(self.config["ui"]["mirror_preview"])
        control_layout.addWidget(self.mirror_checkbox)

        self.debug_checkbox = QCheckBox("إظهار بيانات التتبع المتقدمة (Debug)")
        self.debug_checkbox.setChecked(self.config["ui"]["show_debug"])
        control_layout.addWidget(self.debug_checkbox)

        side_panel.addWidget(control_box)

        diagnostics_box = QGroupBox("التشخيصات المباشرة 📊")
        diagnostics_layout = QVBoxLayout(diagnostics_box)
        self.quality_label = QLabel("جودة التتبع: 0%")
        self.blink_label = QLabel("مؤشر الرمش: n/a")
        self.pose_label = QLabel("وضعية الرأس: yaw 0 | pitch 0 | roll 0")
        self.last_action_label = QLabel("آخر عملية: لا يوجد")
        diagnostics_layout.addWidget(self.quality_label)
        diagnostics_layout.addWidget(self.blink_label)
        diagnostics_layout.addWidget(self.pose_label)
        diagnostics_layout.addWidget(self.last_action_label)
        diagnostics_layout.addWidget(QLabel("تقدم النقر بالتوقف"))
        self.dwell_progress = QProgressBar()
        self.dwell_progress.setRange(0, 100)
        diagnostics_layout.addWidget(self.dwell_progress)
        side_panel.addWidget(diagnostics_box)

        help_box = QGroupBox("خطوات العمل الذكية")
        help_layout = QVBoxLayout(help_box)
        help_layout.addWidget(QLabel("1. قم بفتح الكاميرا أولاً ثم ابدأ المعايرة الذكية"))
        help_layout.addWidget(QLabel("2. حافظ على ثبات نظرك على النقطة المطلوبة"))
        help_layout.addWidget(QLabel("3. انتظر حتى يقوم النظام بقفل النقطة تلقائياً"))
        help_layout.addWidget(QLabel("4. احفظ الإعدادات بعد ضبط الحساسية المناسبة لك"))
        side_panel.addWidget(help_box)
        side_panel.addStretch(1)

    def enqueue_action(self, action, payload=None):
        self.action_queue.put({"action": action, "payload": payload})

    def set_tracking_result(self, tracking_result):
        self.tracking_result = tracking_result

    def set_runtime_state(self, runtime_state):
        self.runtime_state = runtime_state or {}
        self._refresh_runtime_widgets()

    def get_selected_monitor_index(self):
        return self.monitor_combo.currentData()

    def get_calibration_monitor_geometry(self, index):
        app = QtWidgets.QApplication.instance()
        if app is not None:
            screens = app.screens()
            if 0 <= index < len(screens):
                rect = screens[index].availableGeometry()
                return SimpleNamespace(
                    x=rect.x(),
                    y=rect.y(),
                    width=rect.width(),
                    height=rect.height(),
                    name=screens[index].name(),
                )
        monitor = self.monitors[index]
        return SimpleNamespace(
            x=monitor.x,
            y=monitor.y,
            width=monitor.width,
            height=monitor.height,
            name=getattr(monitor, "name", f"Monitor {index + 1}"),
        )

    def get_sensitivity(self):
        slider_value = self.sensitivity_slider.value()
        return 0.55 + (slider_value - 1) * 0.11

    def get_click_mode(self):
        return self.click_mode_combo.currentText()

    def is_mirror_enabled(self):
        return self.mirror_checkbox.isChecked()

    def is_debug_enabled(self):
        return self.debug_checkbox.isChecked()

    def set_click_mode(self, click_mode):
        self.click_mode_combo.blockSignals(True)
        self.click_mode_combo.setCurrentText(click_mode)
        self.click_mode_combo.blockSignals(False)

    def set_rest_mode(self, enabled):
        self.rest_mode_checkbox.blockSignals(True)
        self.rest_mode_checkbox.setChecked(enabled)
        self.rest_mode_checkbox.blockSignals(False)

    def update_preview(self):
        frame = self.camera_manager.get_frame()
        if frame is None:
            return

        tracking_result = self.tracking_result
        mirror_preview = self.is_mirror_enabled()
        display_frame = frame.copy()
        self._draw_overlays(display_frame, tracking_result, mirror_preview)
        if mirror_preview:
            display_frame = cv2.flip(display_frame, 1)

        rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        scaled = QPixmap.fromImage(qt_image).scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def show_calibration_overlay(self, overlay_state):
        if overlay_state is None:
            self.calibration_overlay.hide_overlay()
            return
        self.calibration_overlay.show_target(overlay_state)

    def hide_calibration_overlay(self):
        self.calibration_overlay.hide_overlay()

    def snapshot_settings(self):
        return {
            "click_mode": self.get_click_mode(),
            "sensitivity": self.get_sensitivity(),
            "mirror_preview": self.is_mirror_enabled(),
            "show_debug": self.is_debug_enabled(),
            "selected_monitor_index": self.get_selected_monitor_index(),
            "rest_mode": self.rest_mode_checkbox.isChecked(),
        }

    def _draw_overlays(self, frame, tracking_result, mirror_preview):
        if tracking_result is None:
            return

        x1, y1, x2, y2 = tracking_result["face_bbox"]
        frame_width = frame.shape[1]
        if mirror_preview:
            x1, x2 = frame_width - x2, frame_width - x1

        cv2.rectangle(frame, (x1, y1), (x2, y2), (76, 214, 152), 2)

        for key in ("left", "right"):
            for point in tracking_result["iris_points"][key]:
                px, py = self._display_point(point, frame_width, mirror_preview)
                cv2.circle(frame, (px, py), 2, (56, 189, 248), -1)

        nose_x, nose_y = self._display_point(tracking_result["nose_point"], frame_width, mirror_preview)
        cv2.circle(frame, (nose_x, nose_y), 4, (255, 200, 0), -1)

        if self.is_debug_enabled():
            cv2.putText(
                frame,
                f"Gaze {tracking_result['normalized_point'][0]:.2f}, {tracking_result['normalized_point'][1]:.2f}",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

    def _refresh_runtime_widgets(self):
        state = self.runtime_state
        tracking_text = "نشط" if state.get("face_detected") else "بانتظار الوجه..."
        if not self.camera_manager.active:
            tracking_text = "الكاميرا مغلقة"
            
        self.status_label.setText(f"الحالة: {tracking_text}")
        self.metrics_label.setText(
            f"إطارات: {state.get('fps', 0):.1f} | معالج: {state.get('cpu_percent', 'n/a')}"
        )
        self.calibration_label.setText(state.get("calibration_text", "المعايرة: قيد الانتظار"))
        self.cursor_label.setText(state.get("cursor_text", "موقع الماوس: n/a"))
        self.quality_label.setText(f"جودة التتبع: {int(state.get('tracking_quality', 0.0) * 100)}%")
        self.blink_label.setText(f"مؤشر الرمش: {state.get('blink_ear', 'n/a')}")
        self.pose_label.setText(state.get("pose_text", "وضعية الرأس: n/a"))
        self.last_action_label.setText(f"آخر عملية: {state.get('last_action', 'لا يوجد')}")
        self.dwell_progress.setValue(int(state.get("dwell_progress", 0.0) * 100))

    def _click_mode_changed(self, click_mode):
        mode_map = {
            "رمشة (Blink)": "Blink",
            "توقف (Dwell)": "Dwell",
            "إيقاف (Off)": "Off"
        }
        internal_mode = mode_map.get(click_mode, "Blink")
        self.enqueue_action("set_click_mode", internal_mode)

    def _monitor_changed(self, index):
        self.enqueue_action("set_monitor", self.monitor_combo.itemData(index))

    def _rest_mode_changed(self, state):
        self.enqueue_action("set_rest_mode", state != 0)

    def _sensitivity_to_slider(self, value):
        normalized = max(1, min(15, int(round((value - 0.55) / 0.11 + 1))))
        return normalized

    def _display_point(self, point, frame_width, mirror_preview):
        px, py = int(point[0]), int(point[1])
        if mirror_preview:
            px = frame_width - px
        return px, py
