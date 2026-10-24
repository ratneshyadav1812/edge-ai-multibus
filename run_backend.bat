@echo off
:: Start the backend. Opens http://localhost:8000/ui in browser.
cd /d "%~dp0"
python -m uvicorn backend.server:socket_app --host 0.0.0.0 --port 8000
