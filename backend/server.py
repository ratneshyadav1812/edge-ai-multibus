"""
Backend server — pure event broker.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Urban Road Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

from pydantic import BaseModel

class RoadEvent(BaseModel):
    bus_id: str
    event_type: str
    latitude: float
    longitude: float
    timestamp: str
    details: str
