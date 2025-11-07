
# -*- coding: utf-8 -*-
"""
engine_detect_yolov8.py
───────────────────────────────────────────────
YOLOv8-seg 기반 명함 객체 탐지 + 마스크/세그먼트 저장 엔진
───────────────────────────────────────────────
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from loguru import logger
from ultralytics import YOLO

# ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # /02_ObjectDetection
ROOT_DIR = os.path.dirname(BASE_DIR)                        # /20251107_NameCard
INPUT_DIR = os.path.join(ROOT_DIR, "01_Background/output")  # 증강 이미지 폴더
OUT_DIR = os.path.join(BASE_DIR, "detected")                # 결과 폴더
LOG_DIR = os.path.join(BASE_DIR, "logs")                    # 로그 폴더

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
logger.add(os.path.join(LOG_DIR, "detect_yolov8.log"), level="INFO")

# ───────────────────────────────────────────────
# YOLOv8 Segmentation 모델 로드
# - pretrained: yolov8x-seg.pt (COCO)
# - custom: best.pt (사용자 fine-tuned 모델)
MODEL_PATH = "yolov8x-seg.pt"
model = YOLO(MODEL_PATH)

# ───────────────────────────────────────────────
def save_segments(image_path, results, out_root):
    """YOLOv8 탐지 결과를 개별 crop + mask로 저장 (masks None 예외 처리 포함)"""
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join(out_root, img_name)
    os.makedirs(out_dir, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        return 0

    # 탐지 결과 없을 경우 건너뜀
    if results.masks is None or results.boxes is None or len(results.boxes) == 0:
        logger.warning(f"[{img_name}] No objects detected.")
        return 0

    cv2.imwrite(os.path.join(out_dir, "image.png"), image)

    count = 0
    for box, mask in zip(results.boxes.xyxy, results.masks.data):
        x1, y1, x2, y2 = map(int, box.tolist())
        seg_crop = image[y1:y2, x1:x2]
        if seg_crop.size == 0:
            continue

        mask_np = mask.cpu().numpy().astype(np.uint8)
        mask_np = cv2.resize(mask_np, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST) * 255

        colored_mask = np.zeros_like(seg_crop)
        colored_mask[mask_np > 127] = (0, 255, 0)
        overlay = cv2.addWeighted(seg_crop, 0.8, colored_mask, 0.4, 0)

        cv2.imwrite(os.path.join(out_dir, f"seg_{count+1}.png"), seg_crop)
        cv2.imwrite(os.path.join(out_dir, f"mask_{count+1}.png"), mask_np)
        cv2.imwrite(os.path.join(out_dir, f"overlay_{count+1}.png"), overlay)
        count += 1

    logger.info(f"[{img_name}] Detected {count} card(s).")
    return count


# ───────────────────────────────────────────────
def main():
    images = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".png", ".jpg"))])
    logger.info(f"Found {len(images)} images in {INPUT_DIR}")

    total_cards = 0
    for fn in tqdm(images, desc="🔍 YOLOv8-Seg Detection"):
        image_path = os.path.join(INPUT_DIR, fn)

        # YOLOv8 탐지 수행
        results = model.predict(
            source=image_path,
            conf=0.45,        # 신뢰도 임계값
            iou=0.5,          # NMS IoU 임계값
            imgsz=1280,       # 입력 크기
            device=0 if hasattr(model.model, 'device') else 'cpu',
            verbose=False
        )[0]

        total_cards += save_segments(image_path, results, OUT_DIR)

    logger.info(f"✅ Total detected cards: {total_cards}")

# ───────────────────────────────────────────────
if __name__ == "__main__":
    main()
