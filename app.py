import asyncio
import base64
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEVICE = 0 if torch.cuda.is_available() else "cpu"
SOURCES = {"webcam", "capture"}
CATALOG = {
    "yolo26n": {
        "path": ROOT / "models" / "yolo26n.pt",
        "label": "YOLO26n",
        "imgsz": 640,
        "conf": 0.25,
    },
    "sign": {
        "path": ROOT / "models" / "temp-sign-yolo26n.pt",
        "label": "표지판·YOLO26n",
        "imgsz": 640,
        "conf": 0.35,
    },
    "sign_full": {
        "path": ROOT / "models" / "traffic-sign-yolo11s.pt",
        "label": "표지판82",
        "imgsz": 1280,
        "conf": 0.28,
    },
}
NCNN_DIR = ROOT / "models" / "yolo26n_ncnn_model"
SIGN_KO = {
    "sign": "표지판",
    "No Entry": "진입금지",
    "Turn Right Only": "우회전만",
    "Speed limit 40km/h": "속도제한 40",
    "No Trucks and Bus": "화물차·버스 금지",
    "No Trucks": "화물차 금지",
    "Low Clearance": "높이제한",
    "No Cars": "승용차 금지",
    "Danger": "위험",
    "Slow Down": "서행",
    "Double curve first to right": "우로 굽은 연속커브",
    "Obstacle on the Road": "도로 장애물",
    "Road with Surveillance Camera": "감시카메라",
    "Speed limit 60km/h": "속도제한 60",
    "No Moto": "이륜차 금지",
    "Lane Allocation": "차로지정",
    "Height Limit": "높이제한",
    "No Horns": "경적금지",
    "No Left Turn": "좌회전 금지",
    "No Right Turn": "우회전 금지",
    "No U-Turn": "유턴 금지",
    "No U-Turn and No Left Turn": "유턴·좌회전 금지",
    "No U-Turn and No Right Turn": "유턴·우회전 금지",
    "Traffic light ahead": "신호기",
    "No Stopping & No Parking": "정차·주차 금지",
    "No Parking": "주차 금지",
    "No Straight and Right Turn": "직진·우회전 금지",
    "No Left or Right Turn": "좌우회전 금지",
    "Sharp Left Turn": "좌로 급커브",
    "Sharp Right Turn": "우로 급커브",
    "Intersection with a Minor Road": "소로 교차",
    "Intersection with Equal Roads": "교차로",
    "No Motobike Left Turn": "이륜차 좌회전 금지",
    "Intersection with a Priority Road": "우선도로 교차",
    "Pedestrian Lane": "보행자도로",
    "Narrow Road Left Side": "좌측 좁아짐",
    "Narrow Road Right Side": "우측 좁아짐",
    "Narrow road both sides": "양쪽 좁아짐",
    "No Two or Three-wheeled Vehicles": "이륜·삼륜 금지",
    "Speed Bump": "과속방지턱",
    "Speed limit 50km/h": "속도제한 50",
    "Speed limit 70km/h": "속도제한 70",
    "Speed limit 80km/h": "속도제한 80",
    "Level Crossing with Barriers": "차단기 있는 건널목",
    "No U-Turn for Cars": "자동차 유턴 금지",
    "No bus": "버스 금지",
    "No Overtaking": "추월 금지",
    "Children Crossing": "어린이보호",
    "Pedestrian Crossing": "횡단보도",
    "End of all prohibition": "금지해제",
    "Keep left": "좌측통행",
    "Road Work Ahead": "도로공사",
    "One way street": "일방통행",
    "Turn Left": "좌회전",
    "Turn Right": "우회전",
    "Green Light": "초록불",
    "Red Light": "빨간불",
    "Roundabout": "회전교차로",
    "Speed limit 10km/h": "속도제한 10",
    "Speed limit 100km/h": "속도제한 100",
    "Speed limit 110km/h": "속도제한 110",
    "Speed limit 120km/h": "속도제한 120",
    "Speed limit 20km/h": "속도제한 20",
    "Speed limit 30km/h": "속도제한 30",
    "Speed limit 90km/h": "속도제한 90",
    "Stop": "정지",
    "U-Turn Area": "유턴구간",
    "No Parking on Odd Days": "홀수일 주차금지",
    "No Parking on Even Days": "짝수일 주차금지",
    "Parking": "주차장",
    "Bus Stop": "버스정류장",
    "Hospital": "병원",
    "No U-Turn and Left Turn for Cars": "자동차 유턴·좌회전 금지",
    "Accident area": "사고다발",
    "Dual carriageway": "분리도로",
    "No left turn for cars": "자동차 좌회전 금지",
    "Steep ascent": "오르막",
    "Narrow bridge": "좁은 교량",
    "Uneven road": "노면고르지 않음",
    "End of 50km/h speed limit": "속도제한 50 해제",
    "Residential area": "주거지역",
    "sparsely populated area": "비주거지역",
    "Slippery Road": "미끄러운 도로",
}

app = FastAPI()
loaded = {}
active = "sign_full"
predict_lock = threading.Lock()


def draw_signs(frame, detections):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(r"C:\Windows\Fonts\malgun.ttf", 18)
    for item in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in item["box"]]
        label = f"{item['label']} {item['confidence'] * 100:.0f}%"
        draw.rectangle((x1, y1, x2, y2), outline=(80, 220, 140), width=3)
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_w = right - left
        text_h = bottom - top
        label_top = max(0, y1 - text_h - 8)
        draw.rectangle((x1, label_top, x1 + text_w + 8, label_top + text_h + 6), fill=(20, 48, 36))
        draw.text((x1 + 4, label_top + 2), label, font=font, fill=(217, 255, 233))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def load_models():
    for key, spec in CATALOG.items():
        if not spec["path"].is_file():
            raise FileNotFoundError(f"모델 파일이 없습니다: {spec['path']}")
        loaded[key] = YOLO(str(spec["path"]))


def current_state():
    spec = CATALOG[active]
    return {
        "id": active,
        "model": spec["label"],
        "device": "cuda" if DEVICE == 0 else "cpu",
        "ready": active in loaded,
        "imgsz": spec["imgsz"],
        "ncnn": NCNN_DIR.is_dir(),
        "models": [{"id": key, "label": item["label"]} for key, item in CATALOG.items()],
    }


@app.on_event("startup")
def startup():
    load_models()


@app.post("/api/model")
def choose_model(name: str = Form(...)):
    global active
    if name not in CATALOG:
        raise HTTPException(status_code=400, detail="모델 없음")
    if name not in loaded:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않음")
    active = name
    return current_state()


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/health")
def health():
    return current_state()


@app.post("/api/detect")
async def detect(source: str = Form(...), file: UploadFile = File(...)):
    if source not in SOURCES:
        raise HTTPException(status_code=400, detail="source는 webcam 또는 capture")
    if active not in loaded:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않음")
    spec = CATALOG[active]
    model = loaded[active]

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="이미지가 비어 있음")

    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="이미지를 읽지 못함")

    def infer():
        started = time.perf_counter()
        with predict_lock:
            detected = model.predict(
                frame,
                device=DEVICE,
                verbose=False,
                imgsz=spec["imgsz"],
                conf=spec["conf"],
                iou=0.5,
                max_det=15,
            )[0]
        return detected, (time.perf_counter() - started) * 1000

    result, elapsed_ms = await asyncio.to_thread(infer)

    detections = []
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls.item())
            raw_name = result.names[cls_id]
            detections.append(
                {
                    "label": SIGN_KO.get(raw_name, raw_name),
                    "confidence": round(float(box.conf.item()), 3),
                    "box": [round(v, 1) for v in box.xyxy[0].tolist()],
                }
            )

    image = None
    if source == "capture":
        plotted = draw_signs(frame, detections)
        ok, encoded = cv2.imencode(".jpg", plotted, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise HTTPException(status_code=500, detail="결과 이미지를 만들지 못함")
        image = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")

    height, width = frame.shape[:2]
    return {
        "source": source,
        "width": width,
        "height": height,
        "device": "cuda" if DEVICE == 0 else "cpu",
        "inference_ms": round(elapsed_ms, 1),
        "count": len(detections),
        "detections": detections,
        "image": image,
    }
