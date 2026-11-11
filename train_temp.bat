@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" prepare_temp_dataset.py
".venv\Scripts\yolo.exe" train model=models\yolo26n.pt data=dataset\data.yaml epochs=25 imgsz=640 batch=16 device=0 patience=8 project=runs/detect name=temp_sign exist_ok=True workers=4
copy /Y "runs\detect\runs\detect\temp_sign\weights\best.pt" "models\temp-sign-yolo26n.pt"
pause
