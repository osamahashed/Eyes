import cv2
import threading
import time


class CameraManager:
    def __init__(self, config):
        self.config = config
        self.cap = None
        self.latest_frame = None
        self.latest_timestamp = 0.0
        self.frame_lock = threading.Lock()
        self.active = False
        self.thread = None
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.on_disconnect = None

    def start(self):
        if self.active:
            return
        camera_index = self.config["video"]["camera_index"]
        backend = self._resolve_backend(self.config["video"].get("backend", "default"))
        self.cap = cv2.VideoCapture(camera_index, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["video"]["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["video"]["height"])
        self.cap.set(cv2.CAP_PROP_FPS, self.config["video"]["fps"])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config["video"].get("buffer_size", 1))
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
        self.active = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.active = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        if self.cap:
            self.cap.release()
            self.cap = None
        with self.frame_lock:
            self.latest_frame = None
            
    def is_running(self):
        return self.active

    def get_frame(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def get_latest_timestamp(self):
        with self.frame_lock:
            return self.latest_timestamp

    def _capture_loop(self):
        frame_interval = 1 / max(self.config["video"]["fps"], 1)
        fail_count = 0
        while self.active:
            ret, frame = self.cap.read()
            if ret:
                fail_count = 0
                enhanced_frame = self._enhance_frame(frame)
                with self.frame_lock:
                    self.latest_frame = enhanced_frame
                    self.latest_timestamp = time.perf_counter()
            else:
                fail_count += 1
                if fail_count > 20: # ~2 seconds of missed frames at 10fps retry 
                    self.active = False
                    if self.on_disconnect:
                        self.on_disconnect()
                    break
            time.sleep(frame_interval * 0.35)

    def _enhance_frame(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _resolve_backend(self, backend_name):
        backend_name = str(backend_name).lower()
        if backend_name == "dshow":
            return cv2.CAP_DSHOW
        if backend_name == "msmf":
            return cv2.CAP_MSMF
        return cv2.CAP_ANY
