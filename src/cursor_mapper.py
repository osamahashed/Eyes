import numpy as np
import screeninfo


class ThinPlateSpline:
    def __init__(self, src_pts, dst_pts, reg=1e-3):
        self.src_pts = np.asarray(src_pts, dtype=np.float32)
        self.dst_pts = np.asarray(dst_pts, dtype=np.float32)
        self.num_pts = self.src_pts.shape[0]
        self.reg = float(reg)
        
        # Determine bounds for normalization [0,1]
        self.src_min = np.min(self.src_pts, axis=0)
        self.src_max = np.max(self.src_pts, axis=0) + 1e-6
        self.dst_min = np.min(self.dst_pts, axis=0)
        self.dst_max = np.max(self.dst_pts, axis=0) + 1e-6
        
        src_norm = (self.src_pts - self.src_min) / (self.src_max - self.src_min)
        dst_norm = (self.dst_pts - self.dst_min) / (self.dst_max - self.dst_min)
        
        # Calculate K matrix
        diff = src_norm[:, np.newaxis, :] - src_norm[np.newaxis, :, :]
        r2 = np.sum(diff ** 2, axis=-1)
        # U(r) = r^2 * log(r^2 + epsilon)
        K = r2 * np.log(r2 + 1e-6)
        K += np.eye(self.num_pts) * self.reg
        
        # Calculate P matrix
        P = np.hstack([np.ones((self.num_pts, 1)), src_norm])
        
        # L matrix
        L = np.zeros((self.num_pts + 3, self.num_pts + 3))
        L[:self.num_pts, :self.num_pts] = K
        L[:self.num_pts, self.num_pts:] = P
        L[self.num_pts:, :self.num_pts] = P.T
        
        # Y matrix
        Y = np.zeros((self.num_pts + 3, 2))
        Y[:self.num_pts, :] = dst_norm
        
        # Solve for W
        try:
            self.W = np.linalg.solve(L, Y)
        except np.linalg.LinAlgError:
            self.W = np.linalg.lstsq(L, Y, rcond=None)[0]
            
    def transform(self, pts, k_edge=0.0):
        pts = np.asarray(pts, dtype=np.float32)
        is_single = pts.ndim == 1
        if is_single:
            pts = pts.reshape(1, 2)
            
        pts_norm = (pts - self.src_min) / (self.src_max - self.src_min)
        src_norm = (self.src_pts - self.src_min) / (self.src_max - self.src_min)
        
        diff = pts_norm[:, np.newaxis, :] - src_norm[np.newaxis, :, :]
        r2 = np.sum(diff ** 2, axis=-1)
        U = r2 * np.log(r2 + 1e-6)
        
        P = np.hstack([np.ones((pts.shape[0], 1)), pts_norm])
        
        L_eval = np.hstack([U, P])
        out_norm = L_eval @ self.W
        
        # Apply robust Edge Expansion overcoming splines pull-to-center
        if k_edge > 0.0:
            out_norm = out_norm + k_edge * (out_norm - 0.5)
            
        out = out_norm * (self.dst_max - self.dst_min) + self.dst_min
        return out[0] if is_single else out


class CursorMapper:
    def __init__(self, calibration_data=None, config=None):
        self.config = config or {}
        self.tracking_cfg = self.config.get("tracking", {})
        self.monitors = screeninfo.get_monitors()
        self.monitor_index = min(
            self.config.get("monitors", {}).get("selected_monitor_index", 0),
            max(len(self.monitors) - 1, 0),
        )
        self.tps_model = None
        self.set_calibration(calibration_data)
        
        self.grid_validation_enabled = False
        self.last_grid_state = None

    def set_calibration(self, calibration_data):
        self.calibration = calibration_data
        self.tps_model = None
        
        if self.calibration and self._mapping_name() == "tps_mapping":
            coefs = self.calibration.get("mapping", {}).get("coefficients", {})
            if "gaze_points" in coefs and "screen_points" in coefs:
                try:
                    self.tps_model = ThinPlateSpline(coefs["gaze_points"], coefs["screen_points"])
                except Exception:
                    self.tps_model = None

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
        
        if model_name == "tps_mapping" and self.tps_model is not None:
            gaze = data_vector[2:4] # pure raw_x, raw_y
            
            # Fetch dynamic k_edge multiplier to fight edge-pull
            k_edge = float(self.config.get("calibration", {}).get("edge_boost", 0.05))
            screen_p = self.tps_model.transform(gaze, k_edge=k_edge)
            
            # Head pose assist removed to isolate gaze completely

            
            if getattr(self, "grid_validation_enabled", False):
                self.last_grid_state = {
                    "gaze_input": gaze,
                    "screen_p": screen_p,
                    "barycentric": (0,0,1),
                    "active_triangle": "TPS_MAPPED",
                    "screen_grid": self.tps_model.dst_pts,
                    "gaze_grid": self.tps_model.src_pts,
                    "triangles_topology": []
                }
                
            return screen_p
            
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


