@echo off
:: Run inference on test_video.mp4. Start run_backend.bat first.
cd /d "%~dp0"
python inference.py --video test_video.mp4 --backend http://localhost:8000
