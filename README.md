# Multi-Bus Urban Road Intelligence System

An end-to-end edge-to-dashboard intelligent transportation system that turns transit buses into real-time road condition sensors.

## Architecture

- **Edge Inference Engine (`inference.py`)**: Runs YOLO-based pothole detection and ByteTrack vehicle tracking with synthetic GPS telemetry, debouncing congestion events using `EdgeCongestionFilter`.
- **Event Broker API (`backend/server.py`)**: FastAPI ASGI server with Socket.IO websockets broadcasting telemetry to clients in real-time.
- **Operator Dashboard (`frontend/index.html`)**: Real-time Leaflet map interface with pulsing visual markers, live telemetry feed, and key metrics.
- **Dataset Pipeline (`merge_datasets.py`, `check_dataset.py`)**: Automated dataset unification, SHA-256 deduplication, and sanity checking.

## Quickstart

1. **Setup Environment**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure API Key (Optional for map tiles)**:
   ```powershell
   cp .env.example .env
   # Add your GEOAPIFY_API_KEY if available
   ```

3. **Start Backend**:
   ```powershell
   .\run_backend.bat
   ```
   Open dashboard at `http://localhost:8000/ui`

4. **Run Edge Inference**:
   ```powershell
   .\run_inference.bat
   ```

## Unit Tests
Run the engine unit test suite:
```powershell
python -m unittest discover -s tests
```
