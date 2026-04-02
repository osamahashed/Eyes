import json
import os
import time

import numpy as np


class CalibrationManager:
    CALIBRATION_VERSION = 3

    def __init__(self, config_path, config):
        self.config_path = config_path
        self.config = config
        self.screen_transform = None
        self.active = False
        self.monitor = None
        self.points = []
        self.current_index = 0
        self.point_frame_count = 0
        self.current_samples = []
        self.recent_points = []
        self.calibration_pairs = []
        self.last_quality = 0.0
        self.last_stability = None
        self.last_hint = "Look at the target until lock is stable"

    def begin(self, monitor, monitor_index):
        calibration_cfg = self.config["calibration"]
        self.monitor = {
            "x": monitor.x,
            "y": monitor.y,
            "width": monitor.width,
            "height": monitor.height,
            "index": monitor_index,
        }
        raw_points = self._generate_grid_points(
            self.monitor,
            calibration_cfg["rows"],
            calibration_cfg["cols"],
            calibration_cfg["margin"],
        )
        self.points = self._order_points(raw_points, calibration_cfg.get("order", "center-out"))
        self.current_index = 0
        self.point_frame_count = 0
        self.current_samples = []
        self.recent_points = []
        self.calibration_pairs = []
        self.last_quality = 0.0
        self.last_stability = None
        self.last_hint = "Look at the target until lock is stable"
        self.active = True
        return self.get_overlay_state()

    def observe(self, sample):
        if not self.active:
            return {"status": "idle"}

        self.point_frame_count += 1
        point = np.asarray(sample["normalized_point"], dtype=np.float32)
        quality = float(sample.get("tracking_quality", 0.0))
        head_pose = sample.get("head_pose", {})
        blink_triggered = bool(sample.get("blink_triggered", False))
        yaw = abs(float(head_pose.get("yaw", 0.0)))
        pitch = abs(float(head_pose.get("pitch", 0.0)))

        self.last_quality = quality
        self.recent_points.append(point)
        stable_window = int(self.config["calibration"]["stable_window"])
        self.recent_points = self.recent_points[-stable_window:]
        stability = self._compute_stability(self.recent_points)
        self.last_stability = stability

        settle_frames = int(self.config["calibration"]["settle_frames"])
        samples_per_point = int(self.config["calibration"]["samples_per_point"])
        min_tracking_quality = float(self.config["calibration"]["min_tracking_quality"])
        max_stability = float(self.config["calibration"]["max_stability"])
        max_head_yaw = float(self.config["calibration"]["max_head_yaw"])
        max_head_pitch = float(self.config["calibration"]["max_head_pitch"])

        settled = self.point_frame_count > settle_frames
        stable = stability is not None and stability <= max_stability
        pose_ok = yaw <= max_head_yaw and pitch <= max_head_pitch
        quality_ok = quality >= min_tracking_quality
        ready = settled and stable and pose_ok and quality_ok and not blink_triggered
        self.last_hint = self._build_hint(settled, stable, pose_ok, quality_ok, blink_triggered, quality, stability)

        if ready:
            self.current_samples.append(
                {
                    "normalized_point": point,
                    "tracking_quality": quality,
                    "stability": stability,
                    "yaw": yaw,
                    "pitch": pitch,
                }
            )

        aggregated = None
        if len(self.current_samples) >= samples_per_point:
            aggregated = self._aggregate_samples(self.current_samples)
            # Relaxed threshold for capture
            stability_threshold = max(0.04, float(self.config["calibration"]["max_stability"]) * 2.0)
            min_effective = max(10, samples_per_point // 2)
            if aggregated["effective_samples"] < min_effective or aggregated["stability"] > stability_threshold:
                self.current_samples = []
                aggregated = None
                self.last_hint = "النظرة غير مستقرة، يرجى الهدوء والتركيز قليلاً"

        if aggregated is None:
            return {
                "status": "collecting",
                "progress": min(len(self.current_samples) / max(samples_per_point, 1), 1.0),
                "overlay": self.get_overlay_state(),
            }

        screen_point = np.asarray(self.points[self.current_index], dtype=np.float32)
        self.calibration_pairs.append(
            {
                "screen_point": screen_point.tolist(),
                "gaze_point": aggregated["center"].tolist(),
                "tracking_quality": aggregated["tracking_quality"],
                "stability": aggregated["stability"],
                "effective_samples": aggregated["effective_samples"],
            }
        )

        self.current_index += 1
        self.point_frame_count = 0
        self.current_samples = []
        self.recent_points = []
        self.last_hint = "أحسنت! انتقل بعينيك إلى الهدف التالي..."

        if self.current_index >= len(self.points):
            self._finalize()
            self.active = False
            self.save_calibration()
            return {
                "status": "completed",
                "overlay": None,
                "calibration": self.screen_transform,
            }

        return {
            "status": "advance",
            "progress": 0.0,
            "overlay": self.get_overlay_state(),
        }

    def get_overlay_state(self):
        if not self.active:
            return None
        return {
            "monitor": self.monitor,
            "point": self.points[self.current_index],
            "index": self.current_index + 1,
            "total": len(self.points),
            "progress": min(
                len(self.current_samples) / max(int(self.config["calibration"]["samples_per_point"]), 1),
                1.0,
            ),
            "accepted_samples": len(self.current_samples),
            "target_samples": int(self.config["calibration"]["samples_per_point"]),
            "tracking_quality": self.last_quality,
            "stability": self.last_stability,
            "hint": self.last_hint,
        }

    def clear_calibration(self):
        self.screen_transform = None
        if os.path.exists(self.config_path):
            os.remove(self.config_path)

    def load_calibration(self):
        if not os.path.exists(self.config_path):
            return None
        with open(self.config_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not self._is_compatible(loaded):
            self.screen_transform = None
            return None
        self.screen_transform = loaded
        return self.screen_transform

    def save_calibration(self):
        if self.screen_transform is None:
            return
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.screen_transform, f, indent=2)

    def has_calibration(self):
        return self.screen_transform is not None

    def _generate_grid_points(self, monitor, rows, cols, margin):
        calibration_cfg = self.config["calibration"]
        target_radius = float(calibration_cfg.get("target_radius_px", 18))
        edge_padding = float(calibration_cfg.get("edge_padding_px", 150))
        reference_scale = min(monitor["width"] / 1920.0, monitor["height"] / 1080.0)
        scaled_padding = edge_padding * max(reference_scale, 0.65)

        x_padding = max(monitor["width"] * margin, scaled_padding, target_radius * 4.0)
        y_padding = max(monitor["height"] * margin, scaled_padding, target_radius * 4.0)
        x_padding = min(x_padding, monitor["width"] * 0.22)
        y_padding = min(y_padding, monitor["height"] * 0.22)

        x_min = monitor["x"] + x_padding
        x_max = monitor["x"] + monitor["width"] - x_padding
        y_min = monitor["y"] + y_padding
        y_max = monitor["y"] + monitor["height"] - y_padding

        x_values = np.linspace(x_min, x_max, cols)
        y_values = np.linspace(y_min, y_max, rows)
        points = []
        for y in y_values:
            for x in x_values:
                points.append((float(x), float(y)))
        return points

    def _order_points(self, points, order_name):
        if order_name != "center-out":
            return points
        center = np.mean(np.asarray(points, dtype=np.float32), axis=0)
        return sorted(
            points,
            key=lambda point: (
                np.linalg.norm(np.asarray(point, dtype=np.float32) - center),
                np.arctan2(point[1] - center[1], point[0] - center[0]),
            ),
        )

    def _aggregate_samples(self, samples):
        points = np.array([sample["normalized_point"] for sample in samples], dtype=np.float32)
        center = np.median(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        median_distance = float(np.median(distances))
        mad = float(np.median(np.abs(distances - median_distance)))
        max_stability = float(self.config["calibration"]["max_stability"])
        outlier_scale = float(self.config["calibration"]["outlier_mad_scale"])
        threshold = max(max_stability * 2.5, median_distance + outlier_scale * max(mad, 1e-4))
        inlier_mask = distances <= threshold
        inliers = points[inlier_mask]
        if len(inliers) == 0:
            inliers = points
            inlier_mask = np.ones(len(points), dtype=bool)

        center = np.mean(inliers, axis=0)
        inlier_samples = [sample for sample, keep in zip(samples, inlier_mask) if keep]
        return {
            "center": center,
            "effective_samples": len(inliers),
            "tracking_quality": float(np.mean([sample["tracking_quality"] for sample in inlier_samples])),
            "stability": float(np.mean([sample["stability"] for sample in inlier_samples])),
            "inlier_mask": inlier_mask.tolist(),
        }

    def _finalize(self):
        gaze_points = np.array([pair["gaze_point"] for pair in self.calibration_pairs], dtype=np.float32)
        screen_points = np.array([pair["screen_point"] for pair in self.calibration_pairs], dtype=np.float32)

        models_to_try = self._models_to_try(len(gaze_points))
        fits = [self._fit_model(gaze_points, screen_points, model_name) for model_name in models_to_try]
        
        # Robust selection: Prefer Affine if Quadratic is too erratic (CV error > 1.5x)
        best_fit = fits[0] # Affine is usually first
        if len(fits) > 1:
            affine = fits[0]
            quad = fits[1]
            if quad["cross_validation_error_px"] < affine["cross_validation_error_px"] * 1.5:
                # Only use quadratic if it's significantly better or not much worse (preventing overfitting/jumps)
                best_fit = quad if quad["cross_validation_error_px"] < affine["cross_validation_error_px"] else affine
            else:
                best_fit = affine

        # Accuracy Heatmap Data Generation
        accuracy_report_data = []
        features = self._feature_matrix(gaze_points, best_fit["model_name"])
        predicted = features @ np.array(best_fit["coefficients"])
        errors = np.linalg.norm(predicted - screen_points, axis=1)
        
        for i, error in enumerate(errors):
            color = "green" if error < 70 else "yellow" if error < 140 else "red"
            accuracy_report_data.append({
                "point_index": i,
                "screen_x": float(screen_points[i][0]),
                "screen_y": float(screen_points[i][1]),
                "error_px": float(error),
                "color_classification": color
            })
            
        mean_error = best_fit["mean_error_px"]
        accuracy_score = max(0.0, 100.0 * (1.0 - mean_error / float(self.config["calibration"]["validation_threshold_px"])))

        self.screen_transform = {
            "version": self.CALIBRATION_VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "monitor_index": self.monitor["index"],
            "monitor_bounds": self.monitor,
            "input_transform": self._tracking_signature(),
            "mapping": {
                "name": best_fit["model_name"],
                "coefficients": best_fit["coefficients"],
            },
            "mean_error_px": best_fit["mean_error_px"],
            "max_error_px": best_fit["max_error_px"],
            "cross_validation_error_px": best_fit["cross_validation_error_px"],
            "quality_mean": float(np.mean([pair["tracking_quality"] for pair in self.calibration_pairs])),
            "stability_mean": float(np.mean([pair["stability"] for pair in self.calibration_pairs])),
            "validation_state": "good"
            if best_fit["cross_validation_error_px"] <= float(self.config["calibration"]["validation_threshold_px"])
            else "needs_review",
            "accuracy_report": {
                "points": accuracy_report_data,
                "accuracy_score_percent": float(accuracy_score)
            },
            "pairs": self.calibration_pairs,
        }

    def _models_to_try(self, sample_count):
        preferred = self.config["calibration"].get("preferred_model", "auto")
        if preferred == "affine":
            return ["affine"]
        if preferred == "quadratic":
            return ["quadratic"] if sample_count >= 6 else ["affine"]
        models = ["affine"]
        if sample_count >= 6:
            models.append("quadratic")
        return models

    def _fit_model(self, gaze_points, screen_points, model_name):
        features = self._feature_matrix(gaze_points, model_name)
        coefficients = self._solve_ridge(features, screen_points)
        predicted = features @ coefficients
        errors = np.linalg.norm(predicted - screen_points, axis=1)
        return {
            "model_name": model_name,
            "coefficients": coefficients.tolist(),
            "mean_error_px": float(np.mean(errors)),
            "max_error_px": float(np.max(errors)),
            "cross_validation_error_px": self._cross_validate(gaze_points, screen_points, model_name),
        }

    def _cross_validate(self, gaze_points, screen_points, model_name):
        feature_count = self._feature_matrix(gaze_points[:1], model_name).shape[1]
        if len(gaze_points) <= feature_count:
            coefficients = self._solve_ridge(self._feature_matrix(gaze_points, model_name), screen_points)
            predicted = self._feature_matrix(gaze_points, model_name) @ coefficients
            return float(np.mean(np.linalg.norm(predicted - screen_points, axis=1)))

        errors = []
        for idx in range(len(gaze_points)):
            train_mask = np.ones(len(gaze_points), dtype=bool)
            train_mask[idx] = False
            train_x = gaze_points[train_mask]
            train_y = screen_points[train_mask]
            test_x = gaze_points[~train_mask]
            test_y = screen_points[~train_mask]
            coefficients = self._solve_ridge(self._feature_matrix(train_x, model_name), train_y)
            predicted = self._feature_matrix(test_x, model_name) @ coefficients
            errors.extend(np.linalg.norm(predicted - test_y, axis=1).tolist())
        return float(np.mean(errors))

    def _solve_ridge(self, features, targets):
        ridge = float(self.config["calibration"]["ridge"])
        regularization = np.eye(features.shape[1], dtype=np.float32) * ridge
        regularization[-1, -1] = 0.0
        left = features.T @ features + regularization
        right = features.T @ targets
        return np.linalg.solve(left, right)

    def _feature_matrix(self, points, model_name):
        points = np.asarray(points, dtype=np.float32)
        x = points[:, 0:1]
        y = points[:, 1:2]
        ones = np.ones((len(points), 1), dtype=np.float32)
        if model_name == "quadratic":
            return np.hstack((x, y, x * y, x * x, y * y, ones)).astype(np.float32)
        return np.hstack((x, y, ones)).astype(np.float32)

    def _compute_stability(self, points):
        if len(points) < 2:
            return None
        arr = np.asarray(points, dtype=np.float32)
        center = np.mean(arr, axis=0)
        distances = np.linalg.norm(arr - center, axis=1)
        return float(np.mean(distances))

    def _build_hint(self, settled, stable, pose_ok, quality_ok, blink_triggered, quality, stability):
        if not settled:
            return "استعد... النقطة التالية تظهر الآن"
        if not pose_ok:
            return "حافظ على استقامة رأسك للحصول على أدق النتائج"
        if not quality_ok:
            return "تنبيه: الإضاءة ضعيفة أو المسافة غير مثالية"
        if blink_triggered:
            return "يرجى تجنب الرمش أثناء التقاط العينة الذكية"
        if not stable:
            if stability is None:
                return "ثبّت نظرك على النواة... جاري التحليل"
            return f"ثبّت نظرك... استقرار النظام: {max(0, 100 - stability*2000):.1f}%"
        return "تم القفل! جاري استخراج البيانات الأسطورية..."

    def _tracking_signature(self):
        tracking_cfg = self.config["tracking"]
        calibration_cfg = self.config["calibration"]
        return {
            "invert_x": bool(tracking_cfg.get("invert_x", False)),
            "invert_y": bool(tracking_cfg.get("invert_y", False)),
            "eye_weight": float(tracking_cfg["eye_weight"]),
            "head_weight_x": float(tracking_cfg["head_weight_x"]),
            "head_weight_y": float(tracking_cfg["head_weight_y"]),
            "vertical_gain": float(tracking_cfg["vertical_gain"]),
            "preferred_model": calibration_cfg.get("preferred_model", "auto"),
            "order": calibration_cfg.get("order", "center-out"),
        }

    def _is_compatible(self, calibration_data):
        return (
            calibration_data.get("version", 0) >= self.CALIBRATION_VERSION
            and calibration_data.get("input_transform") == self._tracking_signature()
            and "mapping" in calibration_data
        )
