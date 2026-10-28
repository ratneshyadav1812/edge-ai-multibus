# Multi-Bus Urban Road Intelligence System

An edge-to-cloud transportation intelligence platform that turns fleet buses into real-time road condition sensors. The system detects road surface hazards (potholes) and analyzes traffic congestion locally on edge devices, transmitting telemetry to a centralized real-time map dashboard.

---

## Architecture Flowchart

```mermaid
---
config:
  theme: base
  flowchart:
    curve: basis
    htmlLabels: true
    nodeSpacing: 22
    rankSpacing: 30
    padding: 10
  themeVariables:
    fontFamily: Arial, sans-serif
    fontSize: 12px
    lineColor: "#8293A8"
    textColor: "#EAF1FA"
    clusterBkg: "#101B2B"
    clusterBorder: "#526783"
---
flowchart TB

    subgraph EDGE["01 · EDGE PROCESSING (BUS NODE)"]
        direction TB

        Cam(["Video / Dashcam Stream"])
        Frame["Frame Capture (OpenCV)"]

        subgraph INFERENCE["DUAL MODEL INFERENCE"]
            direction TB
            PotholeDetect["Pothole Detection<br/>pothole_model.pt"]
            VehicleTrack["Vehicle Tracking<br/>yolo11n.pt + ByteTrack"]
        end

        Cam --> Frame
        Frame --> PotholeDetect
        Frame --> VehicleTrack

        AreaFilter{"Area ≥ min_area<br/>(1000px)?"}
        PotholeDetect --> AreaFilter

        PotholeEvent["Pothole Event Payload"]
        DiscardPothole["Ignore Noise"]

        AreaFilter -->|Yes| PotholeEvent
        AreaFilter -->|No| DiscardPothole

        TrafficClass["Traffic Density Classifier<br/>High / Med / Low"]
        CongestionFilter["EdgeCongestionFilter<br/>engine/congestion_filter.py"]
        CongestionCheck{"Sustained ≥ 5s AND<br/>(Cooldown ≥ 15s OR<br/>Dist ≥ 500)?"}

        VehicleTrack --> TrafficClass
        TrafficClass --> CongestionFilter
        CongestionFilter --> CongestionCheck

        CongestionEvent["Congestion Event Payload"]
        DropCongestion["Suppress / Debounce"]

        CongestionCheck -->|Yes| CongestionEvent
        CongestionCheck -->|No| DropCongestion

        Annotator["Visual Annotation<br/>cv2.putText + BBoxes"]
        VideoOut[("Annotated Video<br/>output/annotated_video.mp4")]

        Frame --> Annotator
        Annotator --> VideoOut

        CSVLog[("Local CSV Logs<br/>output/*.csv")]
        Dispatcher["HTTP Telemetry Client<br/>post_event"]

        PotholeEvent --> CSVLog
        CongestionEvent --> CSVLog
        PotholeEvent --> Dispatcher
        CongestionEvent --> Dispatcher
    end

    subgraph BACKEND["02 · CENTRAL BROKER API"]
        direction TB

        FastAPI["FastAPI Ingestion Endpoint<br/>POST /api/events"]
        SocketServer["Socket.IO AsyncServer"]
        StaticMount["Static File Server<br/>/ui"]

        FastAPI --> SocketServer
        FastAPI --> StaticMount
    end

    Dispatcher -->|"HTTP POST"| FastAPI

    subgraph DASHBOARD["03 · OPERATOR DASHBOARD"]
        direction TB

        Browser(["Browser Client<br/>Socket.IO"])

        MapLayer["Leaflet Map<br/>Geoapify / CARTO Dark"]
        Markers["Pulsing Pin Markers<br/>Rose: Pothole<br/>Amber: Congestion"]
        StatCards["Live Metric Counters<br/>Potholes / Congestion"]
        FeedList["Sliding Live Event Feed"]

        Browser --> MapLayer
        MapLayer --> Markers
        Browser --> StatCards
        Browser --> FeedList
    end

    SocketServer -->|"WebSocket:<br/>new_road_event"| Browser
    StaticMount -->|"HTTP GET /ui"| Browser

    classDef source fill:#193A5A,stroke:#60A5FA,color:#F0F7FF,stroke-width:2px
    classDef process fill:#173B43,stroke:#36B8A6,color:#E5FFFA,stroke-width:1.5px
    classDef decision fill:#493A1E,stroke:#EAB308,color:#FFF4C2,stroke-width:2px
    classDef pothole fill:#512B40,stroke:#FB7185,color:#FFF1F4,stroke-width:2px
    classDef congestion fill:#4A351C,stroke:#FBBF24,color:#FFF6DD,stroke-width:2px
    classDef rejected fill:#2A3445,stroke:#738197,color:#D0D9E6,stroke-dasharray:5 4
    classDef storage fill:#25344B,stroke:#8BA5C7,color:#F0F6FF,stroke-width:1.5px
    classDef backend fill:#26345A,stroke:#818CF8,color:#F1F1FF,stroke-width:1.5px
    classDef dashboard fill:#193D35,stroke:#4ADE80,color:#E9FFF0,stroke-width:1.5px

    class Cam source
    class Frame,PotholeDetect,VehicleTrack,TrafficClass,CongestionFilter,Annotator process
    class AreaFilter,CongestionCheck decision
    class PotholeEvent pothole
    class CongestionEvent congestion
    class DiscardPothole,DropCongestion rejected
    class CSVLog,VideoOut storage
    class Dispatcher,FastAPI,SocketServer,StaticMount backend
    class Browser,MapLayer,Markers,StatCards,FeedList dashboard

    style EDGE fill:#101B2B,stroke:#526783,stroke-width:2px,color:#F0F6FF
    style INFERENCE fill:#14263A,stroke:#52718F,stroke-width:1px,color:#EAF1FA
    style BACKEND fill:#121D35,stroke:#818CF8,stroke-width:2px,color:#F0F0FF
    style DASHBOARD fill:#102720,stroke:#4ADE80,stroke-width:2px,color:#E9FFF0
```
---

## Important Modules

| Module | Location | Purpose |
|---|---|---|
| **Edge Inference Pipeline** | [`inference.py`](file:///D:/PotholeProject/inference.py) | Main video processing loop. Runs YOLO pothole inference, YOLO ByteTrack vehicle tracking, synthetic GPS drift modeling, and telemetry dispatch. |
| **Edge Congestion Filter** | [`engine/congestion_filter.py`](file:///D:/PotholeProject/engine/congestion_filter.py) | Temporal debouncer that prevents alert spam. Confirms congestion only if traffic remains high for 5 seconds, subject to a 15-second cooldown or 500-unit displacement trigger. |
| **Event Broker API** | [`backend/server.py`](file:///D:/PotholeProject/backend/server.py) | FastAPI application paired with an ASGI Socket.IO server. Validates telemetry payloads via Pydantic (`RoadEvent`), broadcasts events via WebSockets, and hosts the frontend UI under `/ui`. |
| **Operator Dashboard** | [`frontend/index.html`](file:///D:/PotholeProject/frontend/index.html) | Single-page monitoring interface styled with Tailwind CSS and Leaflet.js. Features live counters, dark-mode mapping with fallback tiles, and animated detection cards. |
| **Dataset Deduplication & Merger** | [`merge_datasets.py`](file:///D:/PotholeProject/merge_datasets.py) | Merges heterogeneous pothole datasets, runs SHA-256 content deduplication, normalizes bounding box annotations, partitions into 80/10/10 splits, and exports YOLO `data.yaml`. |
| **Dataset Sanity Checker** | [`check_dataset.py`](file:///D:/PotholeProject/check_dataset.py) | Samples random images from train/val/test splits and plots bounding box annotations via OpenCV and Matplotlib to verify label alignment. |

---

## Project Setup & Running

This project uses [`uv`](https://github.com/astral-sh/uv), an extremely fast Python package and project manager.

### 1. Prerequisites

- Python 3.10+ installed
- `uv` installed. If you do not have it, install via:
  ```powershell
  # Windows PowerShell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
  *(or via `pip install uv`)*

### 2. Virtual Environment Setup with `uv`

Create and activate a virtual environment using `uv`:

```powershell
# Create virtual environment in .venv
uv venv .venv

# Activate environment (Windows)
.\.venv\Scripts\activate

# (Linux / macOS equivalent: source .venv/bin/activate)
```

### 3. Install Dependencies with `uv`

Install pinned dependencies directly into the environment:

```powershell
uv pip install -r requirements.txt
```

### 4. Configuration

Copy the example environment configuration:

```powershell
cp .env.example .env
```

*(Optional)* Open `.env` and add your [Geoapify API key](https://www.geoapify.com/) for high-resolution dark-matter map tiles. If omitted, the dashboard automatically falls back to CARTO basemaps.

---

## Running the System

### Step 1: Start Backend Server

Launch the event broker and UI server:

```powershell
uv run uvicorn backend.server:socket_app --host 0.0.0.0 --port 8000 --reload
```
*(Or run `.\run_backend.bat`)*

Access the operator dashboard in your browser:
👉 **`http://localhost:8000/ui`**

### Step 2: Run Edge Inference

In a separate terminal, trigger inference on the sample test video:

```powershell
uv run python inference.py --video test_video.mp4 --backend http://localhost:8000 --bus-id BUS-001
```
*(Or run `.\run_inference.bat`)*

**Controls & Outputs:**
- Press `q` in the OpenCV video window to stop inference.
- Annotated video is stored in `output/annotated_video.mp4`.
- Extracted telemetry logs are written to `output/pothole_events.csv` and `output/congestion_events.csv`.

---

## Running Unit Tests

Run the engine test suite verifying congestion filtering, state transitions, and distance overrides:

```powershell
uv run python -m unittest discover -s tests
```
