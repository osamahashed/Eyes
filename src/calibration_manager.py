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
        self.recent_head_poses = []
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
        self.recent_head_poses = []
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
        yaw_signed = float(head_pose.get("yaw", 0.0))
        pitch_signed = float(head_pose.get("pitch", 0.0))
        yaw = abs(yaw_signed)
        pitch = abs(pitch_signed)

        self.last_quality = quality
        self.recent_points.append(point)
        self.recent_head_poses.append([yaw_signed, pitch_signed])
        
        stable_window = int(self.config["calibration"]["stable_window"])
        self.recent_points = self.recent_points[-stable_window:]
        self.recent_head_poses = self.recent_head_poses[-stable_window:]
        
        stability = self._compute_stability(self.recent_points)
        self.last_stability = stability
        
        if len(self.recent_head_poses) > 1:
            head_pts = np.asarray(self.recent_head_poses, dtype=np.float32)
            head_variance = float(np.var(head_pts[:, 0]) + np.var(head_pts[:, 1]))
        else:
            head_variance = 0.0

        settle_frames = int(self.config["calibration"]["settle_frames"])
        samples_per_point_min = int(self.config["calibration"].get("samples_per_point_min", 40))
        samples_per_point_max = int(self.config["calibration"].get("samples_per_point_max", 60))
        min_tracking_quality = float(self.config["calibration"]["min_tracking_quality"])
        max_stability = float(self.config["calibration"]["max_stability"])
        max_head_yaw = float(self.config["calibration"]["max_head_yaw"])
        max_head_pitch = float(self.config["calibration"]["max_head_pitch"])
        max_head_variance = float(self.config["calibration"].get("max_head_variance", 5.0))

        settled = self.point_frame_count > settle_frames
        stable = stability is not None and stability <= max_stability
        head_stable = head_variance <= max_head_variance
        pose_ok = yaw <= max_head_yaw and pitch <= max_head_pitch
        quality_ok = quality >= min_tracking_quality
        is_blinking = sample.get("is_blinking", False)
        # NEVER capture during any blink state (active or settling)
        ready = settled and stable and head_stable and pose_ok and quality_ok and not blink_triggered and not is_blinking
        self.last_hint = self._build_hint(settled, stable, pose_ok, head_stable, quality_ok, blink_triggered or is_blinking, quality, stability)

        if ready:
            self.current_samples.append(
                {
                    "normalized_point": point,
                    "raw_gaze_ratio": sample.get("raw_gaze_ratio"),
                    "tracking_quality": quality,
                    "stability": stability,
                    "yaw": yaw_signed,
                    "pitch": pitch_signed,
                    "raw_ear": sample.get("raw_ear", 1.0),
                }
            )

        aggregated = None
        current_len = len(self.current_samples)
        
        if current_len >= samples_per_point_min:
            aggregated = self._aggregate_samples(self.current_samples)
            # High quality: high effective samples count and highly stable cluster
            is_high_quality = aggregated["effective_samples"] >= (samples_per_point_min * 0.85) and aggregated["stability"] <= max_stability
            
            if is_high_quality or current_len >= samples_per_point_max:
                # Decide if we accept it or fail it
                stability_threshold = max(0.04, max_stability * 2.0)
                min_effective = max(10, current_len // 2)
                
                if aggregated["effective_samples"] < min_effective or aggregated["stability"] > stability_threshold:
                    self.current_samples = []
                    aggregated = None
                    self.last_hint = "ركز نظرك على الهدف بثبات"
            else:
                # Keep holding for more frames
                aggregated = None

        if aggregated is None:
            return {
                "status": "collecting",
                "progress": min(len(self.current_samples) / max(samples_per_point_max, 1), 1.0),
                "overlay": self.get_overlay_state(),
            }

        screen_point = np.asarray(self.points[self.current_index], dtype=np.float32)
        self.calibration_pairs.append(
            {
                "screen_point": screen_point.tolist(),
                "gaze_point": aggregated["center"].tolist(),
                "raw_gaze_ratio": aggregated["raw_gaze_ratio"].tolist() if isinstance(aggregated["raw_gaze_ratio"], np.ndarray) else list(aggregated["raw_gaze_ratio"]),
                "yaw": aggregated["yaw"],
                "pitch": aggregated["pitch"],
                "tracking_quality": aggregated["tracking_quality"],
                "stability": aggregated["stability"],
                "effective_samples": aggregated["effective_samples"],
                "raw_ear": aggregated["raw_ear"],
            }
        )

        self.current_index += 1
        self.point_frame_count = 0
        self.current_samples = []
        self.recent_points = []
        self.recent_head_poses = []
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
                len(self.current_samples) / max(int(self.config["calibration"].get("samples_per_point_max", 60)), 1),
                1.0,
            ),
            "accepted_samples": len(self.current_samples),
            "target_samples": int(self.config["calibration"].get("samples_per_point_max", 60)),
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
        center = np.median(points, axis=0) # Measure from median for robustness
        distances = np.linalg.norm(points - center, axis=1)
        mean_dist = float(np.mean(distances))
        std_dist = float(np.std(distances))
        
        # Outlier filtering beyond 1.5 std deviations (stricter rejection for noisy inputs)
        threshold = max(0.01, mean_dist + 1.5 * std_dist)
        inlier_mask = distances <= threshold
        inliers = points[inlier_mask]
        
        if len(inliers) == 0:
            inliers = points
            inlier_mask = np.ones(len(points), dtype=bool)

        center = np.median(inliers, axis=0) # Readjust median
        inlier_samples = [sample for sample, keep in zip(samples, inlier_mask) if keep]
        
        raw_gaze_ratios = [s["raw_gaze_ratio"] for s in inlier_samples if s["raw_gaze_ratio"] is not None]
        avg_raw_gaze = np.median(raw_gaze_ratios, axis=0) if raw_gaze_ratios else center
        
        return {
            "center": center,
            "raw_gaze_ratio": avg_raw_gaze,
            "yaw": float(np.mean([s["yaw"] for s in inlier_samples])),
            "pitch": float(np.mean([s["pitch"] for s in inlier_samples])),
            "effective_samples": len(inliers),
            "tracking_quality": float(np.mean([sample["tracking_quality"] for sample in inlier_samples])),
            "stability": float(np.mean([sample["stability"] for sample in inlier_samples])),
            "raw_ear": float(np.mean([sample.get("raw_ear", 1.0) for sample in inlier_samples])),
            "inlier_mask": inlier_mask.tolist(),
        }

    def _finalize(self):
        # Ensure strict row-by-row, column-by-column sorting for the 9-point grid
        self.calibration_pairs = sorted(self.calibration_pairs, key=lambda p: (float(p["screen_point"][1]), float(p["screen_point"][0])))
        
        gaze_points = np.array([
            [
                pair["gaze_point"][0], 
                pair["gaze_point"][1],
                pair.get("raw_gaze_ratio", pair["gaze_point"])[0],
                pair.get("raw_gaze_ratio", pair["gaze_point"])[1],
                pair.get("yaw", 0.0),
                pair.get("pitch", 0.0)
            ]
            for pair in self.calibration_pairs
        ], dtype=np.float32)
        screen_points = np.array([pair["screen_point"] for pair in self.calibration_pairs], dtype=np.float32)

        models_to_try = self._models_to_try(len(gaze_points))
        if "tps_mapping" in models_to_try and len(gaze_points) >= 16:
            best_fit = {
                "model_name": "tps_mapping",
                "coefficients": {
                    "gaze_points": [p.get("raw_gaze_ratio", p["gaze_point"]) for p in self.calibration_pairs],
                    "screen_points": [p["screen_point"] for p in self.calibration_pairs],
                },
                "mean_error_px": 0.0,
                "max_error_px": 0.0,
                "cross_validation_error_px": 0.0,
            }
        else:
            fits = [self._fit_model(gaze_points, screen_points, model_name) for model_name in models_to_try]
            
            # Robust selection: We compare models and choose the best one
            best_fit = fits[0] 
            if len(fits) > 1:
                best_error = fits[0]["cross_validation_error_px"]
                for fit in fits[1:]:
                    # We give affine a slight advantage, quadratic/compensated goes if significantly better
                    coeff_penalty = 1.0
                    if fit["model_name"] == "quadratic":
                        coeff_penalty = 1.5
                    elif fit["model_name"] == "affine_pose_compensated":
                        coeff_penalty = 1.1 # Prefer it if it's 10% better or more to avoid overfitting flat poses
                        
                    if fit["cross_validation_error_px"] < best_error * (2.0 - coeff_penalty):
                        best_fit = fit
                        best_error = fit["cross_validation_error_px"]

        # Accuracy Heatmap Data Generation
        accuracy_report_data = []
        if best_fit["model_name"] == "tps_mapping":
            for i, pair in enumerate(self.calibration_pairs):
                accuracy_report_data.append({
                    "point_index": i,
                    "screen_x": float(pair["screen_point"][0]),
                    "screen_y": float(pair["screen_point"][1]),
                    "error_px": 0.0,
                    "color_classification": "green"
                })
            mean_error = 0.0
            accuracy_score = 100.0
        else:
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
            "baseline_ear": float(np.mean([pair.get("raw_ear", 1.0) for pair in self.calibration_pairs])),
            "accuracy_report": {
                "points": accuracy_report_data,
                "accuracy_score_percent": float(accuracy_score)
            },
            "pairs": self.calibration_pairs,
        }

    def _models_to_try(self, sample_count):
        preferred = self.config["calibration"].get("preferred_model", "auto")
        if preferred == "tps_mapping" and sample_count >= 16:
            return ["tps_mapping"]
        if preferred == "affine":
            return ["affine"]
        if preferred == "affine_pose_compensated":
            return ["affine_pose_compensated"]
        if preferred == "quadratic":
            return ["quadratic"] if sample_count >= 6 else ["affine"]
        
        # default "auto" explores affine vs affine_pose_compensated
        models = ["affine", "affine_pose_compensated"]
        if sample_count >= 6:
            models.append("quadratic")
        return models

    def _compute_coefficients(self, features, screen_points, model_name):
        coefficients = np.zeros((features.shape[1], 2), dtype=np.float32)
        if model_name == "affine_pose_compensated":
            idx_x = [0, 2, 4]  # raw_x, norm_yaw, 1.0
            idx_y = [1, 3, 4]  # raw_y, norm_pitch, 1.0
        elif model_name == "affine":
            idx_x = [0, 2]     # x, 1.0
            idx_y = [1, 2]     # y, 1.0
        elif model_name == "quadratic":
            idx_x = [0, 3, 5]  # x, x*x, 1.0
            idx_y = [1, 4, 5]  # y, y*y, 1.0
        else:
            return self._solve_ridge(features, screen_points)
            
        coefficients[idx_x, 0] = self._solve_ridge(features[:, idx_x], screen_points[:, 0])
        coefficients[idx_y, 1] = self._solve_ridge(features[:, idx_y], screen_points[:, 1])
        return coefficients

    def _fit_model(self, gaze_points, screen_points, model_name):
        features = self._feature_matrix(gaze_points, model_name)
        coefficients = self._compute_coefficients(features, screen_points, model_name)
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
            coefficients = self._compute_coefficients(self._feature_matrix(gaze_points, model_name), screen_points, model_name)
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
            coefficients = self._compute_coefficients(self._feature_matrix(train_x, model_name), train_y, model_name)
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
        x = points[:, 0:1] # blended_x
        y = points[:, 1:2] # blended_y
        raw_x = points[:, 2:3]
        raw_y = points[:, 3:4]
        yaw = points[:, 4:5]
        pitch = points[:, 5:6]
        
        ones = np.ones((len(points), 1), dtype=np.float32)
        if model_name == "quadratic":
            return np.hstack((x, y, x * y, x * x, y * y, ones)).astype(np.float32)
        elif model_name == "affine_pose_compensated":
            norm_yaw = yaw / 45.0
            norm_pitch = pitch / 45.0
            return np.hstack((raw_x, raw_y, norm_yaw, norm_pitch, ones)).astype(np.float32)
            
        return np.hstack((x, y, ones)).astype(np.float32)

    def _compute_stability(self, points):
        if len(points) < 2:
            return None
        arr = np.asarray(points, dtype=np.float32)
        center = np.mean(arr, axis=0)
        distances = np.linalg.norm(arr - center, axis=1)
        return float(np.mean(distances))

    def _build_hint(self, settled, stable, pose_ok, head_stable, quality_ok, blink_triggered, quality, stability):
        if not settled:
            return "استعد"
        if blink_triggered:
            return "تجنب الرمش المفرط الآن"
        if not head_stable or not pose_ok:
            return "ثبت رأسك في المنتصف"
        if not stable:
            return "ركز نظرك على الهدف بثبات"
        return "يتم الحفظ..."

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
