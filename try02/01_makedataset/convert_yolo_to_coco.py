# -*- coding: utf-8 -*-
"""
convert_yolo_to_coco.py
───────────────────────────────────────────────
YOLO-format dataset → COCO-format annotations (instance segmentation)
(파일명 패턴 자동 탐색 + 상세 로그 버전)
───────────────────────────────────────────────
"""
import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from loguru import logger

# ───────────────────────────────────────────────
ROOT = "./output"
SPLITS = ["train", "val", "test"]
COCO_DIR = os.path.join(ROOT, "coco_annotations")
os.makedirs(COCO_DIR, exist_ok=True)

logger.add(os.path.join(COCO_DIR, "convert_to_coco.log"), level="INFO")

CATEGORIES = [{"id": 1, "name": "card"}]

# ───────────────────────────────────────────────
def get_image_info(img_path, img_id):
    img = cv2.imread(img_path)
    if img is None:
        logger.warning(f"⚠️ Cannot read image: {img_path}")
        return None
    h, w = img.shape[:2]
    return {"id": img_id, "file_name": os.path.basename(img_path),
            "height": h, "width": w}

def mask_to_polygons(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        poly = cnt.flatten().tolist()
        if len(poly) >= 6:  # 최소 3개 점
            polygons.append(poly)
    return polygons

def get_annotation_info(mask, img_id, ann_id):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    area = float(cv2.contourArea(cnt))
    segmentation = mask_to_polygons(mask)
    return {
        "id": ann_id,
        "image_id": img_id,
        "category_id": 1,
        "bbox": [x, y, w, h],
        "area": area,
        "segmentation": segmentation,
        "iscrowd": 0
    }

def find_mask(mask_dir, img_name):
    """mask 파일명을 자동으로 매칭"""
    base = os.path.splitext(img_name)[0]
    candidates = [
        os.path.join(mask_dir, f"{base}_mask.png"),
        os.path.join(mask_dir, f"{base.replace('test_', 'test_mask_')}.png"),
        os.path.join(mask_dir, f"{base.replace('train_', 'train_mask_')}.png"),
        os.path.join(mask_dir, f"{base.replace('val_', 'val_mask_')}.png"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

# ───────────────────────────────────────────────
def convert_split(split):
    logger.info(f"🚀 Converting split: {split}")
    img_dir = os.path.join(ROOT, split, "images")
    mask_dir = os.path.join(ROOT, split, "masks")

    coco_json = {
        "info": {"description": "NameCard dataset", "version": "1.0"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": CATEGORIES
    }

    img_id, ann_id = 1, 1
    images = sorted([f for f in os.listdir(img_dir) if f.endswith((".png", ".jpg"))])
    logger.info(f"Found {len(images)} images in {img_dir}")

    for img_name in tqdm(images, desc=f"[{split}]"):
        img_path = os.path.join(img_dir, img_name)
        mask_path = find_mask(mask_dir, img_name)

        if not mask_path:
            logger.warning(f"❌ No matching mask for {img_name}")
            continue

        img_info = get_image_info(img_path, img_id)
        if not img_info:
            continue

        coco_json["images"].append(img_info)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.warning(f"⚠️ Cannot read mask: {mask_path}")
            continue

        ann = get_annotation_info(mask, img_id, ann_id)
        if ann:
            coco_json["annotations"].append(ann)
            ann_id += 1
        else:
            logger.warning(f"⚠️ No valid contour found for {mask_path}")
        img_id += 1

    out_path = os.path.join(COCO_DIR, f"instances_{split}.json")
    with open(out_path, "w") as f:
        json.dump(coco_json, f, indent=2)
    logger.info(f"✅ Saved {out_path}: {len(coco_json['images'])} images, {len(coco_json['annotations'])} annotations")

# ───────────────────────────────────────────────
if __name__ == "__main__":
    for s in SPLITS:
        convert_split(s)
    logger.info("🎯 Conversion to COCO format completed.")


