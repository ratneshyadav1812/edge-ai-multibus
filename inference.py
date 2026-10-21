"""
Inference entry-point.
Runs YOLO pothole + vehicle detection on a video, shows frames in real-time,
and POSTs events to the backend server.
"""
import cv2
from ultralytics import YOLO

def run(video_path: str = "test_video.mp4", min_pothole_area: int = 1000) -> None:
    print("Loading models...")
    pothole_model = YOLO("pothole_model.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pothole_res = pothole_model(frame, verbose=False)[0]

        for box in pothole_res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = (x2 - x1) * (y2 - y1)
            if area < min_pothole_area:
                continue

        cv2.imshow("Urban Road Intelligence", cv2.resize(frame, (800, 450)))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
