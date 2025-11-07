# -*- coding: utf-8 -*-
"""
convert_masks_to_yolo_seg_labels.py
───────────────────────────────────────────────
Mask 이미지를 YOLOv8 Segmentation 라벨로 변환
───────────────────────────────────────────────
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

ROOT = "./output"
SPLITS = ["train", "val", "test"]

def mask_to_yolo_format(mask_path, save_path, class_id=0):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False

    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    lines = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 10:
            continue
        norm = cnt.reshape(-1, 2) / np.array([[w, h]])
        coords = " ".join([f"{x:.6f} {y:.6f}" for x, y in norm])
        lines.append(f"{class_id} {coords}")

    if not lines:
        return False

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    return True


def convert_split(split):
    img_dir = os.path.join(ROOT, split, "images")
    mask_dir = os.path.join(ROOT, split, "masks")
    label_dir = os.path.join(ROOT, split, "labels")
    os.makedirs(label_dir, exist_ok=True)

    images = sorted([f for f in os.listdir(img_dir) if f.endswith((".png", ".jpg"))])
    count = 0

    for img_name in tqdm(images, desc=f"[{split}] converting"):
        base = os.path.splitext(img_name)[0]
        mask_path = os.path.join(mask_dir, f"{split}_mask_{base.split('_')[-1]}.png")
        label_path = os.path.join(label_dir, f"{base}.txt")

        if os.path.exists(mask_path):
            ok = mask_to_yolo_format(mask_path, label_path)
            if ok:
                count += 1

    print(f"✅ {split}: {count}/{len(images)} masks converted.")


if __name__ == "__main__":
    for s in SPLITS:
        convert_split(s)
    print("🎯 YOLOv8 segmentation labels generated successfully.")

