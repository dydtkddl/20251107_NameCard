# -*- coding: utf-8 -*-
"""
vis.py
───────────────────────────────────────────────
YOLOv8 segmentation 결과 시각화 (confidence 자동 제거 포함)
───────────────────────────────────────────────
"""

import os
import cv2
import numpy as np
import random
import logging
from tqdm import tqdm

LABEL_DIR = "./labels"     # YOLO polygon .txt 라벨 경로
IMAGE_DIR = "./"           # 원본 이미지 경로
OUT_DIR = "./visualized"   # 시각화 결과 저장 경로
os.makedirs(OUT_DIR, exist_ok=True)

# ───────────────────────────────
# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("visualize.log", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# ───────────────────────────────
def load_yolo_polygon(label_path, img_w, img_h):
    """
    YOLO polygon segmentation 라벨 로드
    """
    polygons = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 6:
                logging.warning(f"[SKIP] {label_path}: 점 좌표 부족 ({len(parts)})")
                continue

            coords = np.array(parts[1:], dtype=float)

            # (1) confidence 제거
            if len(coords) % 2 == 1:
                coords = coords[:-1]
                logging.debug(f"[INFO] {label_path}: 마지막 confidence 값 제거")

            # (2) 좌표 짝이 안 맞을 경우 버리기
            if len(coords) < 6:
                logging.warning(f"[SKIP] {label_path}: 좌표 수 부족 ({len(coords)})")
                continue

            # (3) reshape 안전 처리
            try:
                coords = coords.reshape(-1, 2)
            except Exception as e:
                logging.error(f"[ERR] reshape 실패 ({label_path}): {e}")
                continue

            abs_coords = np.stack([coords[:, 0] * img_w, coords[:, 1] * img_h], axis=1).astype(np.int32)
            if len(abs_coords) >= 3:
                polygons.append(abs_coords)
            else:
                logging.warning(f"[SKIP] {label_path}: polygon 점 3개 미만")

    return polygons

# ───────────────────────────────
def visualize_segmentation(image_path, label_path, out_path):
    img = cv2.imread(image_path)
    if img is None:
        logging.error(f"[X] 이미지 로드 실패: {image_path}")
        return False

    h, w = img.shape[:2]
    polygons = load_yolo_polygon(label_path, w, h)
    if not polygons:
        logging.warning(f"[ ] polygon 없음: {label_path}")
        return False

    overlay = img.copy()
    for poly in polygons:
        color = [random.randint(60, 230) for _ in range(3)]
        cv2.fillPoly(overlay, [poly], color)
        cv2.polylines(img, [poly], isClosed=True, color=(0, 0, 255), thickness=2)

    vis = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    cv2.imwrite(out_path, vis)
    logging.info(f"[OK] {os.path.basename(out_path)} 저장 완료")
    return True

# ───────────────────────────────
def main():
    label_files = sorted([f for f in os.listdir(LABEL_DIR) if f.endswith(".txt")])
    if not label_files:
        logging.error("라벨 파일이 없습니다.")
        return

    for fn in tqdm(label_files, desc="🎨 Visualizing", ncols=100):
        label_path = os.path.join(LABEL_DIR, fn)
        img_path = os.path.join(IMAGE_DIR, fn.replace(".txt", ".jpg"))
        out_path = os.path.join(OUT_DIR, fn.replace(".txt", "_vis.jpg"))

        if not os.path.exists(img_path):
            img_path = img_path.replace(".jpg", ".png")
            if not os.path.exists(img_path):
                logging.warning(f"[ ] 이미지 없음: {img_path}")
                continue

        visualize_segmentation(img_path, label_path, out_path)

    logging.info(f"\n✅ 시각화 완료 → {OUT_DIR}")

# ───────────────────────────────
if __name__ == "__main__":
    main()
