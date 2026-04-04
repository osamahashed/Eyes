import numpy as np
import screeninfo


class CursorMapper:
    def __init__(self, calibration_data=None, config=None):
        self.config = config or {}
        self.tracking_cfg = self.config.get("tracking", {})
        self.monitors = screeninfo.get_monitors()
        self.monitor_index = min(
            self.config.get("monitors", {}).get("selected_monitor_index", 0),
            max(len(self.monitors) - 1, 0),
        )
        self.calibration = calibration_data
        
        # Adaptive Bias Tracker
        self.gaze_history = []
        self.fixation_anchor = None

    def set_calibration(self, calibration_data):
        self.calibration = calibration_data

    def set_monitor(self, index):
        self.monitor_index = max(0, min(index, len(self.monitors) - 1))

    def get_monitor(self, index=None):
        return self.monitors[self.monitor_index if index is None else index]

    def map_gaze_to_screen(self, tracking_result):
        if tracking_result is None:
            return None
        monitor = self.get_monitor()

        if isinstance(tracking_result, np.ndarray) or isinstance(tracking_result, list):
            # Backward compatibility or raw vector fallback
            point = np.asarray(tracking_result, dtype=np.float32)[:2]
            raw_gaze = point
            yaw, pitch = 0.0, 0.0
        else:
            point = np.asarray(tracking_result.get("normalized_point", [0, 0]), dtype=np.float32)[:2]
            raw_gaze = np.asarray(tracking_result.get("raw_gaze_ratio", point), dtype=np.float32)[:2]
            yaw = float(tracking_result.get("head_pose", {}).get("yaw", 0.0))
            pitch = float(tracking_result.get("head_pose", {}).get("pitch", 0.0))
            
        data_vector = np.array([point[0], point[1], raw_gaze[0], raw_gaze[1], yaw, pitch], dtype=np.float32)

        if self._calibration_matches_monitor():
            screen_point = self._predict_calibrated(data_vector)
            
            # Adaptive Bias Correction (Fixation Anchor)
            # If the raw raw_gaze (pure eye movement) is extremely stable, anchor the cursor.
            self.gaze_history.append(raw_gaze)
            if len(self.gaze_history) > 12:
                self.gaze_history.pop(0)
                
            if len(self.gaze_history) == 12:
                hist = np.asarray(self.gaze_history)
                variance = np.var(hist[:, 0]) + np.var(hist[:, 1])
                
                # If variance is very low, the eye is fixating
                if variance < 0.00005: 
                    if self.fixation_anchor is None:
                        self.fixation_anchor = screen_point
                    else:
                        # Apply bias correction: pull raw output towards the stable anchor
                        screen_point = screen_point * 0.1 + self.fixation_anchor * 0.9
                else:
                    # Release anchor gracefully if eye moves
                    self.fixation_anchor = None
            
            return self._clamp_to_monitor(screen_point, monitor)
        return self._map_raw_to_monitor(point, monitor)

    def calibration_status(self):
        if not self.calibration:
            return {"active": False, "error_px": None, "model_name": None, "validation_state": None}
        return {
            "active": self._calibration_matches_monitor(),
            "error_px": self.calibration.get("mean_error_px"),
            "monitor_index": self.calibration.get("monitor_index"),
            "model_name": self._mapping_name(),
            "validation_state": self.calibration.get("validation_state"),
            "cross_validation_error_px": self.calibration.get("cross_validation_error_px"),
        }

    def _map_raw_to_monitor(self, point, monitor):
        margin = float(self.tracking_cfg.get("screen_margin", 0.08))
        x = self._remap_with_margin(point[0], margin)
        y = self._remap_with_margin(point[1], margin)
        return np.array(
            [
                monitor.x + x * monitor.width,
                monitor.y + y * monitor.height,
            ],
            dtype=np.float32,
        )

    def _clamp_to_monitor(self, point, monitor):
        x = np.clip(point[0], monitor.x, monitor.x + monitor.width - 1)
        y = np.clip(point[1], monitor.y, monitor.y + monitor.height - 1)
        return np.array([x, y], dtype=np.float32)

    def _calibration_matches_monitor(self):
        return (
            self.calibration is not None
            and self.calibration.get("monitor_index") == self.monitor_index
            and self._mapping_coefficients() is not None
        )

    def _remap_with_margin(self, value, margin):
        value = float(np.clip(value, 0.0, 1.0))
        margin = float(np.clip(margin, 0.0, 0.3))
        usable = max(1.0 - margin * 2.0, 1e-6)
        return np.clip((value - margin) / usable, 0.0, 1.0)

    def _predict_calibrated(self, data_vector):
        model_name = self._mapping_name()
        
        if model_name == "grid_mapping":
            gaze = data_vector[:2] # blended_x, blended_y
            mapping = self.calibration["mapping"]["coefficients"]
            return self._interpolate_grid(gaze, mapping["gaze_grid"], mapping["screen_grid"])
            
        coefficients = np.asarray(self._mapping_coefficients(), dtype=np.float32)
        features = self._feature_vector(data_vector, model_name)
        return features @ coefficients

    def _mapping_name(self):
        mapping = self.calibration.get("mapping") if self.calibration else None
        if mapping and "name" in mapping:
            return mapping["name"]
        if self.calibration and "affine_matrix" in self.calibration:
            return "affine"
        return None

    def _mapping_coefficients(self):
        mapping = self.calibration.get("mapping") if self.calibration else None
        if mapping and "coefficients" in mapping:
            return mapping["coefficients"]
        if self.calibration and "affine_matrix" in self.calibration:
            return self.calibration["affine_matrix"]
        return None

    def _feature_vector(self, data_vector, model_name):
        x = float(data_vector[0])
        y = float(data_vector[1])
        raw_x = float(data_vector[2])
        raw_y = float(data_vector[3])
        norm_yaw = float(data_vector[4]) / 45.0
        norm_pitch = float(data_vector[5]) / 45.0
        
        if model_name == "quadratic":
            return np.array([x, y, x * y, x * x, y * y, 1.0], dtype=np.float32)
        elif model_name == "affine_pose_compensated":
            return np.array([raw_x, raw_y, norm_yaw, norm_pitch, 1.0], dtype=np.float32)
            
        return np.array([x, y, 1.0], dtype=np.float32)

    def _interpolate_grid(self, gaze_point, gaze_grid, screen_grid):
        gaze_point = np.asarray(gaze_point, dtype=np.float32)
        gaze_grid = np.asarray(gaze_grid, dtype=np.float32)
        screen_grid = np.asarray(screen_grid, dtype=np.float32)
        
        # Predefined triangle topology forcing consistent diagonals (0-4, 1-5, 3-7, 4-8)
        # Assuming grid is strictly row-major exactly as 0,1,2, 3,4,5, 6,7,8
        triangles = [
            (0, 1, 4), (0, 4, 3), # Top-Left
            (1, 2, 5), (1, 5, 4), # Top-Right
            (3, 4, 7), (3, 7, 6), # Bottom-Left
            (4, 5, 8), (4, 8, 7)  # Bottom-Right
        ]
        
        best_triangle_idx = -1
        best_bary = None
        max_min_bary = -float('inf')
        
        for idx_A, idx_B, idx_C in triangles:
            A = gaze_grid[idx_A]
            B = gaze_grid[idx_B]
            C = gaze_grid[idx_C]
            
            # Compute Barycentric coordinates
            v0 = B - A
            v1 = C - A
            v2 = gaze_point - A
            
            d00 = np.dot(v0, v0)
            d01 = np.dot(v0, v1)
            d11 = np.dot(v1, v1)
            d20 = np.dot(v2, v0)
            d21 = np.dot(v2, v1)
            
            denom = d00 * d11 - d01 * d01
            if abs(denom) < 1e-6:
                continue
                
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1.0 - v - w
            
            min_bary = min(u, v, w)
            if min_bary >= 0:
                # Point is strictly inside this triangle
                best_triangle_idx = (idx_A, idx_B, idx_C)
                best_bary = (u, v, w)
                break
                
            # Track the closest triangle for out-of-bounds extrapolation
            if min_bary > max_min_bary:
                max_min_bary = min_bary
                best_triangle_idx = (idx_A, idx_B, idx_C)
                best_bary = (u, v, w)
                
        if best_triangle_idx == -1:
            # Fallback safety
            return np.array([gaze_point[0]*1920, gaze_point[1]*1080], dtype=np.float32)
            
        u, v, w = best_bary
        idx_A, idx_B, idx_C = best_triangle_idx
        
        screen_p = (
            u * screen_grid[idx_A] +
            v * screen_grid[idx_B] +
            w * screen_grid[idx_C]
        )
        return screen_p
