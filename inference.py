"""
Inference entry-point.
Runs YOLO pothole + vehicle detection on a video, shows frames in real-time,
and POSTs events to the backend server.
"""
import csv
import os
import random
from datetime import datetime
import cv2
from ultralytics import YOLO

def run(video_path: str = "test_video.mp4", min_pothole_area: int = 1000, bus_id: str = "BUS-001") -> None:
    print("Loading models...")
    pothole_model = YOLO("pothole_model.pt")
    vehicle_model = YOLO("yolo11n.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    os.makedirs("output", exist_ok=True)
    with open("output/pothole_events.csv", "w", newline="") as pf, \
         open("output/congestion_events.csv", "w", newline="") as cf:

        pw = csv.writer(pf)
        pw.writerow(["Timestamp", "Bus_ID", "Event_Type", "Latitude", "Longitude", "Pothole_Size_px"])
        cw = csv.writer(cf)
        cw.writerow(["Timestamp", "Bus_ID", "Event_Type", "Latitude", "Longitude", "Vehicle_Count", "Congestion_Status"])

        lat, lon = 29.9457, 76.8116

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pothole_res = pothole_model(frame, verbose=False)[0]
            vehicle_res = vehicle_model.track(
                frame, classes=[2, 3, 5, 7], persist=True,
                tracker="bytetrack.yaml", verbose=False,
            )[0]

            n_vehicles = len(vehicle_res.boxes)
            traffic    = "High" if n_vehicles >= 5 else "Medium" if n_vehicles >= 2 else "Low"

            if traffic == "Low":
                lat += 0.0002; lon += 0.0002
            else:
                lat += random.uniform(0.00001, 0.00005)
                lon += random.uniform(0.00001, 0.00005)

            for box in pothole_res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area = (x2 - x1) * (y2 - y1)
                if area < min_pothole_area:
                    continue
                p_lat = round(lat + random.uniform(-0.0001, 0.0001), 6)
                p_lon = round(lon + random.uniform(-0.0001, 0.0001), 6)
                size  = "Large" if area > 8000 else "Medium" if area > 3000 else "Small"
                details = f"{size} ({int(area)}px)"
                pw.writerow([ts, bus_id, "POTHOLE", p_lat, p_lon, details])
                pf.flush()

            cv2.imshow("Urban Road Intelligence", cv2.resize(frame, (800, 450)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
