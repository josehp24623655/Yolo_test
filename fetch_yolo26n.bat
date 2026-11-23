@echo off
cd /d "%~dp0\models"
"..\.venv\Scripts\python.exe" -c "from ultralytics import YOLO; m=YOLO('yolo26n.pt'); m.export(format='ncnn', imgsz=640)"
pause