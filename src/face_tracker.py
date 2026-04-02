import cv2
import numpy as np

class FaceTracker:
    def __init__(self):
        self.prev_faces = []

    def track_faces(self, detections, frame):
        current_faces = []
        for det in detections:
            bbox = det['bbox']
            cropped = frame[int(bbox[1]*frame.shape[0]):int(bbox[3]*frame.shape[0]), 
                           int(bbox[0]*frame.shape[1]):int(bbox[2]*frame.shape[1])]
            current_faces.append({'bbox': bbox, 'cropped': cropped})
        self.prev_faces = current_faces
        return current_faces