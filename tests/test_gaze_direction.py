from gaze_estimator import GazeEstimator


def test_horizontal_direction_is_natural_when_invert_x_enabled():
    estimator = GazeEstimator.__new__(GazeEstimator)
    estimator.tracking_cfg = {
        "eye_weight": 1.35,
        "head_weight_x": 0.95,
        "head_weight_y": 0.85,
        "vertical_gain": 1.2,
        "invert_x": True,
        "invert_y": False,
    }

    looking_right = estimator._blend_gaze_and_head([0.2, 0.5], -0.1, 0.0)
    looking_left = estimator._blend_gaze_and_head([0.8, 0.5], 0.1, 0.0)

    assert looking_right[0] > 0.5
    assert looking_left[0] < 0.5
