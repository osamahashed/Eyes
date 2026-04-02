import cv2
import numpy as np

def check_cam(index, backend, backend_name):
    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print(f"[{index}:{backend_name}] Failed to open.")
        return
        
    ret, frame = cap.read()
    if not ret:
        print(f"[{index}:{backend_name}] Opened, but failed to read frame.")
    else:
        avg = np.mean(frame)
        print(f"[{index}:{backend_name}] Success: {frame.shape}, Avg brightness: {avg:.2f}")
        if avg < 1.0:
            print(f"[{index}:{backend_name}] WARNING: Frame is completely pitch black!")
    cap.release()

if __name__ == "__main__":
    check_cam(0, cv2.CAP_ANY, "ANY")
    check_cam(0, cv2.CAP_DSHOW, "DSHOW")
    check_cam(1, cv2.CAP_ANY, "ANY")
    check_cam(1, cv2.CAP_DSHOW, "DSHOW")
