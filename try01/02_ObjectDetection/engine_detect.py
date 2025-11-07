
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
def save_segments(image_path, results, out_root):
    """YOLOv8 탐지 결과를 mask 기반으로 정확히 crop + class filter + 여백 제거"""
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join(out_root, img_name)
    os.makedirs(out_dir, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        return 0

    # 탐지 결과 없을 경우
    if results.masks is None or results.boxes is None or len(results.boxes) == 0:
        logger.warning(f"[{img_name}] No objects detected.")
        return 0

    cv2.imwrite(os.path.join(out_dir, "image.png"), image)

    count = 0
    h_img, w_img = image.shape[:2]

    # ① class filter — COCO에서 "book", "tv", "laptop"만 명함 유사체로 유지
    valid_classes = ["book", "tv", "laptop", "cell phone", "remote"]
    names = results.names

    for box, mask, cls_id in zip(results.boxes.xyxy, results.masks.data, results.boxes.cls):
        cls_name = names[int(cls_id)]
        if cls_name not in valid_classes:
            continue  # 엉뚱한 객체 무시

        # ② mask 기반 crop (여백 제거)
        mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
        ys, xs = np.where(mask_np > 128)
        if len(xs) == 0 or len(ys) == 0:
            continue

        # mask 범위로 crop box 계산
        x1, x2 = np.min(xs), np.max(xs)
        y1, y2 = np.min(ys), np.max(ys)

        # 안전 범위 보정
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        seg_crop = image[y1:y2, x1:x2]

        # 너무 작으면 skip
        if seg_crop.shape[0] < 50 or seg_crop.shape[1] < 80:
            continue

        # ③ 마스크 오버레이
        mask_resized = mask_np[y1:y2, x1:x2]
        colored_mask = np.zeros_like(seg_crop)
        colored_mask[mask_resized > 127] = (0, 255, 0)
        overlay = cv2.addWeighted(seg_crop, 0.8, colored_mask, 0.4, 0)

        # 저장
        cv2.imwrite(os.path.join(out_dir, f"seg_{count+1}.png"), seg_crop)
        cv2.imwrite(os.path.join(out_dir, f"mask_{count+1}.png"), mask_resized)
        cv2.imwrite(os.path.join(out_dir, f"overlay_{count+1}.png"), overlay)
        count += 1

    logger.info(f"[{img_name}] Detected {count} card-like objects.")
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
