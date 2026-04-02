import numpy as np

class LandmarkHeadPose:
    def __init__(self):
        pass

    def extract_landmarks(self, model_output):
        # Assume model_output is dict with 'landmarks' key
        landmarks = model_output['landmarks'][0]
        return landmarks

    def extract_head_pose(self, model_output):
        # Assume model_output is dict with 'angle_r_fc', 'angle_p_fc', 'angle_y_fc'
        yaw = model_output['angle_y_fc'][0][0]
        pitch = model_output['angle_p_fc'][0][0]
        roll = model_output['angle_r_fc'][0][0]
        return {'yaw': yaw, 'pitch': pitch, 'roll': roll}