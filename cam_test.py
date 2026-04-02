import cv2

def test_camera():
    print("Testing cameras...")
    for index in range(3):
        print(f"Testing Index {index} with ANY backend...")
        cap = cv2.VideoCapture(index, cv2.CAP_ANY)
        if cap.isOpened():
            ret, frame = cap.read()
            print(f"  [{index}:ANY] isOpened: {cap.isOpened()}, read(): {ret}")
            cap.release()
        else:
            print(f"  [{index}:ANY] isOpened: False")
            
        print(f"Testing Index {index} with DSHOW backend...")
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            print(f"  [{index}:DSHOW] isOpened: {cap.isOpened()}, read(): {ret}")
            cap.release()
        else:
            print(f"  [{index}:DSHOW] isOpened: False")

if __name__ == "__main__":
    test_camera()
