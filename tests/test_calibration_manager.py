import numpy as np

from calibration_manager import CalibrationManager
from config import Config


class DummyMonitor:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height


def test_calibration_finalize_creates_smart_mapping(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    config = Config(str(config_path))
    manager = CalibrationManager(str(tmp_path / "calibration.json"), config.data)
    monitor = DummyMonitor(0, 0, 1920, 1080)
    manager.begin(monitor, 0)

    manager.calibration_pairs = [
        {"gaze_point": [0.15, 0.20], "screen_point": [180.0, 180.0], "tracking_quality": 0.8, "stability": 0.01},
        {"gaze_point": [0.50, 0.20], "screen_point": [960.0, 180.0], "tracking_quality": 0.82, "stability": 0.01},
        {"gaze_point": [0.85, 0.20], "screen_point": [1740.0, 180.0], "tracking_quality": 0.85, "stability": 0.01},
        {"gaze_point": [0.15, 0.50], "screen_point": [180.0, 540.0], "tracking_quality": 0.84, "stability": 0.01},
        {"gaze_point": [0.50, 0.50], "screen_point": [960.0, 540.0], "tracking_quality": 0.88, "stability": 0.01},
        {"gaze_point": [0.85, 0.50], "screen_point": [1740.0, 540.0], "tracking_quality": 0.86, "stability": 0.01},
        {"gaze_point": [0.15, 0.80], "screen_point": [180.0, 920.0], "tracking_quality": 0.83, "stability": 0.01},
        {"gaze_point": [0.50, 0.80], "screen_point": [960.0, 920.0], "tracking_quality": 0.87, "stability": 0.01},
        {"gaze_point": [0.85, 0.80], "screen_point": [1740.0, 920.0], "tracking_quality": 0.84, "stability": 0.01},
    ]

    manager._finalize()

    assert manager.screen_transform["mapping"]["name"] in {"affine", "quadratic"}
    assert "coefficients" in manager.screen_transform["mapping"]
    assert manager.screen_transform["quality_mean"] > 0.8


def test_aggregate_samples_rejects_large_outlier(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    config = Config(str(config_path))
    manager = CalibrationManager(str(tmp_path / "calibration.json"), config.data)

    samples = [
        {"normalized_point": np.array([0.50, 0.50], dtype=np.float32), "tracking_quality": 0.8, "stability": 0.01},
        {"normalized_point": np.array([0.51, 0.49], dtype=np.float32), "tracking_quality": 0.82, "stability": 0.01},
        {"normalized_point": np.array([0.49, 0.51], dtype=np.float32), "tracking_quality": 0.79, "stability": 0.01},
        {"normalized_point": np.array([0.90, 0.10], dtype=np.float32), "tracking_quality": 0.7, "stability": 0.05},
    ]

    aggregated = manager._aggregate_samples(samples)

    assert aggregated["effective_samples"] == 3
    assert np.allclose(aggregated["center"], np.array([0.5, 0.5]), atol=0.03)


def test_generated_points_stay_inside_safe_visible_area(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    config = Config(str(config_path))
    manager = CalibrationManager(str(tmp_path / "calibration.json"), config.data)
    monitor = {"x": 0, "y": 0, "width": 1920, "height": 1080}

    points = manager._generate_grid_points(monitor, 3, 3, config.data["calibration"]["margin"])

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    assert min(xs) > 0
    assert min(ys) > 0
    assert max(xs) < monitor["width"]
    assert max(ys) < monitor["height"]
    assert (max(xs) - min(xs)) < monitor["width"]
    assert (max(ys) - min(ys)) < monitor["height"]
