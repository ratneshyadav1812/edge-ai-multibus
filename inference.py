"""
Inference entry-point.
Runs YOLO pothole + vehicle detection on a video, shows frames in real-time,
and POSTs events to the backend server (which pushes them to the browser).

Usage (from project root):
    python inference.py
    python inference.py --video path/to/video.mp4 --backend http://localhost:8000
"""
import argparse
import csv
import os
import random
import requests
from datetime import datetime

import cv2
from ultralytics import YOLO

from engine.congestion_filter import EdgeCongestionFilter


def post_event(backend_url: str, payload: dict) -> None:
    """Fire-and-forget POST to backend; silently drops on network error."""
    try:
        requests.post(f"{backend_url}/api/events", json=payload, timeout=0.5)
    except requests.exceptions.RequestException:
        pass


def run(video_path: str, backend_url: str, bus_id: str = "BUS-001",
        min_pothole_area: int = 1000) -> None:
    print("Loading models...")
    pothole_model = YOLO("pothole_model.pt")
    vehicle_model = YOLO("yolo11n.pt")
    congestion_manager = EdgeCongestionFilter(sustain_time=5, cooldown_time=15, dist_threshold=500)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs("output", exist_ok=True)
    out = cv2.VideoWriter(
        "output/annotated_video.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height),
    )

    with open("output/pothole_events.csv", "w", newline="") as pf, \
         open("output/congestion_events.csv", "w", newline="") as cf:

        pw = csv.writer(pf)
        pw.writerow(["Timestamp", "Bus_ID", "Event_Type", "Latitude", "Longitude", "Pothole_Size_px"])
        cw = csv.writer(cf)
        cw.writerow(["Timestamp", "Bus_ID", "Event_Type", "Latitude", "Longitude", "Vehicle_Count", "Congestion_Status"])

        print(f"Processing '{video_path}'... Press 'q' to quit.")
        frame_count = 0
        lat, lon = 29.9457, 76.8116  # synthetic start coords

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            t   = frame_count / fps
            ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            pothole_res = pothole_model(frame, verbose=False)[0]
            vehicle_res = vehicle_model.track(
                frame, classes=[2, 3, 5, 7], persist=True,
                tracker="bytetrack.yaml", verbose=False,
            )[0]

            n_vehicles = len(vehicle_res.boxes)
            traffic    = "High" if n_vehicles >= 5 else "Medium" if n_vehicles >= 2 else "Low"

            # Synthetic GPS drift
            if traffic == "Low":
                lat += 0.0002; lon += 0.0002
            else:
                lat += random.uniform(0.00001, 0.00005)
                lon += random.uniform(0.00001, 0.00005)

            # Pothole events
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
                post_event(backend_url, {
                    "bus_id": bus_id, "event_type": "POTHOLE",
                    "latitude": p_lat, "longitude": p_lon,
                    "timestamp": ts, "details": details,
                })

            # Congestion events
            if congestion_manager.process_frame(traffic, t, lat, lon):
                details = f"Traffic: {traffic}, Vehicles: {n_vehicles}"
                cw.writerow([ts, bus_id, "CONGESTION", round(lat, 6), round(lon, 6), n_vehicles, traffic])
                cf.flush()
                print(f"[{ts}] CONGESTION @ {lat:.5f}, {lon:.5f}")
                post_event(backend_url, {
                    "bus_id": bus_id, "event_type": "CONGESTION",
                    "latitude": round(lat, 6), "longitude": round(lon, 6),
                    "timestamp": ts, "details": details,
                })

            # Annotate and display
            annotated = pothole_res.plot()
            annotated = vehicle_res.plot(img=annotated)
            color = (0, 0, 255) if traffic == "High" else (0, 255, 255)
            cv2.putText(annotated, f"Vehicles: {n_vehicles} | Traffic: {traffic}",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            out.write(annotated)
            cv2.imshow("Urban Road Intelligence", cv2.resize(annotated, (800, 450)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Done. Outputs saved to output/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run road intelligence inference")
    parser.add_argument("--video",   default="test_video.mp4",        help="Input video path")
    parser.add_argument("--backend", default="http://localhost:8000",  help="Backend server URL")
    parser.add_argument("--bus-id",  default="BUS-001",                help="Bus identifier")
    args = parser.parse_args()
    run(args.video, args.backend, args.bus_id)
