import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
CACHE = ROOT / "dataset_cache"
COCO_ZIP = CACHE / "coco128.zip"
BUS_URL = "https://ultralytics.com/images/bus.jpg"
TEACHER = ROOT / "models" / "traffic-sign-yolo11s.pt"


def download_file(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 1024:
        return target
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response:
        target.write_bytes(response.read())
    return target


def load_sign_crops():
    bus_path = CACHE / "bus.jpg"
    download_file(BUS_URL, bus_path)
    image = cv2.imread(str(bus_path))
    if image is None:
        raise RuntimeError("bus.jpg 읽기 실패")
    model = YOLO(str(TEACHER))
    result = model.predict(image, conf=0.25, imgsz=960, verbose=False)[0]
    h, w = image.shape[:2]
    crops = []
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            pad = 6
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            crop = image[y1:y2, x1:x2]
            if crop.size:
                crops.append(crop)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 24 or bh < 24 or bw > w * 0.4 or bh > h * 0.4:
            continue
        crop = image[y : y + bh, x : x + bw]
        ratio = bw / bh
        if 0.6 <= ratio <= 1.5:
            crops.append(crop)
    if not crops:
        raise RuntimeError("표지판 크롭 없음")
    return crops


def load_backgrounds():
    download_file(
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip",
        COCO_ZIP,
    )
    extract_dir = CACHE / "coco128"
    if not extract_dir.exists():
        with zipfile.ZipFile(COCO_ZIP, "r") as zf:
            zf.extractall(CACHE)
    images = list((extract_dir / "images" / "train2017").glob("*.jpg"))
    if len(images) < 20:
        raise RuntimeError("배경 이미지 부족")
    return images


def write_split(name, count, backgrounds, crops):
    img_dir = DATASET / "images" / name
    lbl_dir = DATASET / "labels" / name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    for index in range(count):
        bg = cv2.imread(str(random.choice(backgrounds)))
        if bg is None:
            continue
        h, w = bg.shape[:2]
        sign = random.choice(crops).copy()
        scale = random.uniform(0.08, 0.22)
        size = max(24, int(min(h, w) * scale))
        sign = cv2.resize(sign, (size, size))
        if random.random() < 0.5:
            sign = cv2.convertScaleAbs(sign, alpha=random.uniform(0.8, 1.2), beta=random.randint(-20, 20))
        x1 = random.randint(0, max(0, w - size - 1))
        y1 = random.randint(0, max(0, h - size - 1))
        x2 = x1 + size
        y2 = y1 + size
        roi = bg[y1:y2, x1:x2]
        if roi.shape[:2] != sign.shape[:2]:
            continue
        alpha = random.uniform(0.82, 1.0)
        cv2.addWeighted(sign, alpha, roi, 1 - alpha, 0, roi)
        bg[y1:y2, x1:x2] = roi

        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        cv2.imwrite(str(img_dir / f"{name}_{index:04d}.jpg"), bg)
        (lbl_dir / f"{name}_{index:04d}.txt").write_text(
            f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}",
            encoding="utf-8",
        )


def write_yaml():
    yaml = DATASET / "data.yaml"
    yaml.write_text(
        "\n".join(
            [
                f"path: {DATASET.as_posix()}",
                "train: images/train",
                "val: images/val",
                "nc: 1",
                "names:",
                "  0: sign",
            ]
        ),
        encoding="utf-8",
    )
    return yaml


def main():
    if DATASET.exists():
        shutil.rmtree(DATASET)
    backgrounds = load_backgrounds()
    crops = load_sign_crops()
    write_split("train", 160, backgrounds, crops)
    write_split("val", 40, backgrounds, crops)
    yaml = write_yaml()
    train_count = len(list((DATASET / "images" / "train").glob("*.jpg")))
    val_count = len(list((DATASET / "images" / "val").glob("*.jpg")))
    print("train", train_count, "val", val_count, "yaml", yaml)


if __name__ == "__main__":
    main()
