"""
Backend server — pure event broker.
Receives POSTed events from inference.py and broadcasts to browser clients via Socket.IO.
No model/inference logic here.
"""
import os
import pathlib
import socketio
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()  # loads .env from cwd (project root)

app = FastAPI(title="Urban Road Intelligence API")
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend — path is relative to this file so it works from any cwd
_FRONTEND = pathlib.Path(__file__).parent.parent / "frontend"
app.mount("/ui", StaticFiles(directory=str(_FRONTEND), html=True), name="frontend")


class RoadEvent(BaseModel):
    bus_id: str
    event_type: str
    latitude: float
    longitude: float
    timestamp: str
    details: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/config")
async def get_config():
    """Expose public config to the frontend. Geoapify key is a map tile key (visible in browser network tab anyway)."""
    return {"geoapify_api_key": os.getenv("GEOAPIFY_API_KEY", "")}


@app.post("/api/events")
async def receive_event(event: RoadEvent):
    """Inference client POSTs here; we push to all connected browsers instantly."""
    await sio.emit("new_road_event", event.model_dump())
    print(f"[EVENT] {event.event_type} @ {event.latitude:.5f}, {event.longitude:.5f}")
    return {"status": "success"}


@sio.event
async def connect(sid, environ):
    print(f"[WS] connected: {sid}")


@sio.event
async def disconnect(sid):
    print(f"[WS] disconnected: {sid}")


if __name__ == "__main__":
    uvicorn.run("backend.server:socket_app", host="0.0.0.0", port=8000, reload=False)
