from openvino.runtime import Core, AsyncInferQueue
import numpy as np
import threading
from queue import Queue

class ModelRunner:
    def __init__(self, model_paths):
        self.core = Core()
        self.models = {}
        self.async_queues = {}
        self.input_queues = {}
        self.output_queues = {}
        for name, path in model_paths.items():
            model = self.core.read_model(path)
            compiled_model = self.core.compile_model(model, "CPU")
            self.models[name] = compiled_model
            self.async_queues[name] = AsyncInferQueue(compiled_model, 2)
            self.input_queues[name] = Queue()
            self.output_queues[name] = Queue()
            self.async_queues[name].set_callback(self._callback_factory(name))

    def _callback_factory(self, name):
        def callback(request, userdata):
            results = request.results
            self.output_queues[name].put((results, userdata))
        return callback

    def enqueue_inference(self, model_name, inputs, userdata=None):
        self.input_queues[model_name].put((inputs, userdata))

    def get_results(self, model_name):
        return self.output_queues[model_name].get() if not self.output_queues[model_name].empty() else None

    def run_pipeline(self, frame):
        # Face detection
        face_inputs = {'data': frame}
        self.enqueue_inference('face_detection', face_inputs, frame)

        # Process face detection results
        face_result = self.get_results('face_detection')
        if face_result:
            detections, original_frame = face_result
            faces = self._process_face_detections(detections)
            for face in faces:
                # Landmarks
                landmark_inputs = {'data': face['cropped']}
                self.enqueue_inference('landmarks', landmark_inputs, (face, original_frame))

        # Process landmarks results
        landmark_result = self.get_results('landmarks')
        if landmark_result:
            landmarks, (face, original_frame) = landmark_result
            # Head pose
            head_inputs = {'data': face['cropped']}
            self.enqueue_inference('head_pose', head_inputs, (landmarks, face, original_frame))

        # Process head pose results
        head_result = self.get_results('head_pose')
        if head_result:
            head_pose, (landmarks, face, original_frame) = head_result
            # Gaze estimation
            left_eye = face['left_eye']
            right_eye = face['right_eye']
            gaze_inputs = {
                'left_eye_image': left_eye,
                'right_eye_image': right_eye,
                'head_pose_angles': head_pose
            }
            self.enqueue_inference('gaze', gaze_inputs, (landmarks, head_pose, original_frame))

        # Process gaze results
        gaze_result = self.get_results('gaze')
        if gaze_result:
            gaze_vector, (landmarks, head_pose, original_frame) = gaze_result
            return {
                'landmarks': landmarks,
                'head_pose': head_pose,
                'gaze': gaze_vector,
                'frame': original_frame
            }
        return None

    def _process_face_detections(self, detections):
        # Simplified processing; in real implementation, parse detections properly
        faces = []
        # Assume detections is a dict with 'detection_out' key
        dets = detections['detection_out'][0][0]
        for det in dets:
            if det[2] > 0.5:  # confidence
                x1, y1, x2, y2 = det[3:7]
                faces.append({'bbox': (x1, y1, x2, y2), 'cropped': None})  # Placeholder
        return faces