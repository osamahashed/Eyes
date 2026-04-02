import cv2
import mediapipe as mp
import numpy as np


class GazeEstimator:
    LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]
    LEFT_EYE_BOUNDS = {"left": 33, "right": 133, "top": 159, "bottom": 145}
    RIGHT_EYE_BOUNDS = {"left": 362, "right": 263, "top": 386, "bottom": 374}
    LEFT_IRIS = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]
    FACE_BOX_POINTS = [10, 152, 33, 263]
    NOSE_TIP = 1
    FOREHEAD = 10
    CHIN = 152

    def __init__(self, config):
        self.config = config
        self.tracking_cfg = config["tracking"]
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.tracking_cfg["face_detection_confidence"],
            min_tracking_confidence=self.tracking_cfg["face_tracking_confidence"],
        )

    def close(self):
        self.face_mesh.close()

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None

        h, w = frame.shape[:2]
        normalized_points = np.array(
            [[landmark.x, landmark.y, landmark.z] for landmark in result.multi_face_landmarks[0].landmark],
            dtype=np.float32,
        )
        pixel_points = np.column_stack((normalized_points[:, 0] * w, normalized_points[:, 1] * h)).astype(np.float32)

        left_eye = pixel_points[self.LEFT_EYE_EAR]
        right_eye = pixel_points[self.RIGHT_EYE_EAR]
        left_iris = pixel_points[self.LEFT_IRIS]
        right_iris = pixel_points[self.RIGHT_IRIS]

        left_eye_ratio = self._iris_ratio(normalized_points, self.LEFT_IRIS, self.LEFT_EYE_BOUNDS)
        right_eye_ratio = self._iris_ratio(normalized_points, self.RIGHT_IRIS, self.RIGHT_EYE_BOUNDS)
        gaze_ratio = (left_eye_ratio + right_eye_ratio) / 2.0

        nose = normalized_points[self.NOSE_TIP][:2]
        forehead = normalized_points[self.FOREHEAD][:2]
        chin = normalized_points[self.CHIN][:2]
        face_left = normalized_points[33][0]
        face_right = normalized_points[263][0]

        face_width = max(face_right - face_left, 1e-6)
        face_height = max(chin[1] - forehead[1], 1e-6)
        head_center_x = (face_left + face_right) / 2.0
        head_center_y = (forehead[1] + chin[1]) / 2.0
        horizontal_sign = -1.0 if self.tracking_cfg.get("invert_x", False) else 1.0
        vertical_sign = -1.0 if self.tracking_cfg.get("invert_y", False) else 1.0

        head_offset_x = (nose[0] - head_center_x) / face_width
        head_offset_y = (nose[1] - head_center_y) / face_height
        blended_point = self._blend_gaze_and_head(gaze_ratio, head_offset_x, head_offset_y)

        bbox = self._face_bbox(pixel_points[self.FACE_BOX_POINTS], w, h)
        area_ratio = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / max(w * h, 1), 0.0)

        head_pose = {
            "yaw": float(head_offset_x * 120.0 * horizontal_sign),
            "pitch": float(head_offset_y * 120.0 * vertical_sign),
            "roll": float(np.degrees(np.arctan2(normalized_points[263][1] - normalized_points[33][1], normalized_points[263][0] - normalized_points[33][0]))),
        }

        return {
            "normalized_point": blended_point,
            "raw_gaze_ratio": gaze_ratio,
            "head_pose": head_pose,
            "blink_landmarks": {"left_eye": left_eye, "right_eye": right_eye},
            "face_bbox": bbox,
            "iris_points": {"left": left_iris, "right": right_iris},
            "nose_point": np.array([nose[0] * w, nose[1] * h], dtype=np.float32),
            "face_center": np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0], dtype=np.float32),
            "tracking_quality": float(np.clip(area_ratio * 8.0, 0.0, 1.0)),
        }

    def _iris_ratio(self, normalized_points, iris_indices, eye_bounds):
        iris_center = np.mean(normalized_points[iris_indices, :2], axis=0)
        left_corner = normalized_points[eye_bounds["left"], :2]
        right_corner = normalized_points[eye_bounds["right"], :2]
        top_point = normalized_points[eye_bounds["top"], :2]
        bottom_point = normalized_points[eye_bounds["bottom"], :2]

        min_x = min(left_corner[0], right_corner[0])
        max_x = max(left_corner[0], right_corner[0])
        min_y = min(top_point[1], bottom_point[1])
        max_y = max(top_point[1], bottom_point[1])

        ratio_x = self._normalize(iris_center[0], min_x, max_x)
        ratio_y = self._normalize(iris_center[1], min_y, max_y)
        return np.array([ratio_x, ratio_y], dtype=np.float32)

    def _blend_gaze_and_head(self, gaze_ratio, head_offset_x, head_offset_y):
        eye_weight = float(self.tracking_cfg["eye_weight"])
        head_weight_x = float(self.tracking_cfg["head_weight_x"])
        head_weight_y = float(self.tracking_cfg["head_weight_y"])
        vertical_gain = float(self.tracking_cfg["vertical_gain"])
        horizontal_sign = -1.0 if self.tracking_cfg.get("invert_x", False) else 1.0
        vertical_sign = -1.0 if self.tracking_cfg.get("invert_y", False) else 1.0

        x_offset = (gaze_ratio[0] - 0.5) * eye_weight + head_offset_x * head_weight_x
        y_offset = (gaze_ratio[1] - 0.5) * eye_weight * vertical_gain + head_offset_y * head_weight_y
        x = 0.5 + x_offset * horizontal_sign
        y = 0.5 + y_offset * vertical_sign
        return np.array([self._clamp(x), self._clamp(y)], dtype=np.float32)

    def _face_bbox(self, points, width, height):
        min_xy = np.maximum(np.min(points, axis=0), 0)
        max_xy = np.minimum(np.max(points, axis=0), np.array([width - 1, height - 1], dtype=np.float32))
        return (int(min_xy[0]), int(min_xy[1]), int(max_xy[0]), int(max_xy[1]))

    def _normalize(self, value, lower, upper):
        span = max(upper - lower, 1e-6)
        return self._clamp((value - lower) / span)

    def _clamp(self, value):
        return float(np.clip(value, 0.0, 1.0))
