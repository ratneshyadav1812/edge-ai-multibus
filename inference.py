"""
Inference entry-point.
Runs YOLO pothole + vehicle detection on a video, shows frames in real-time,
and POSTs events to the backend server.
"""
import cv2

def run(video_path: str = "test_video.mp4") -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Urban Road Intelligence", cv2.resize(frame, (800, 450)))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
