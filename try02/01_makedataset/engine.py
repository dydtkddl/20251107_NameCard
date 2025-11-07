# -*- coding: utf-8 -*-
"""
engine_makedataset_split.py
───────────────────────────────────────────────
명함 합성 데이터셋 생성 + train/val/test 자동 분할
───────────────────────────────────────────────
"""

import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from loguru import logger
from albumentations import Compose, SafeRotate, ShiftScaleRotate, RandomBrightnessContrast

# ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
BG_DIR = os.path.join(ROOT_DIR, "00_IMGS/Backgrounds/auto_backgrounds")
CARD_DIR = os.path.join(ROOT_DIR, "00_IMGS/NameCards")

OUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(BASE_DIR, "logs")
for d in [OUT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

logger.add(os.path.join(LOG_DIR, "makedataset_split.log"), level="INFO")

# ───────────────────────────────────────────────
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}

card_aug = Compose([
    SafeRotate(limit=10, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=0.9),
    ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=10,
                     border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=0.9),
    RandomBrightnessContrast(0.05, 0.05, p=0.3),
])

# ───────────────────────────────────────────────
def load_images(folder):
    exts = [".jpg", ".png", ".jpeg"]
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in exts]

def check_overlap(new_box, existing_boxes, margin=15):
    x1, y1, x2, y2 = new_box
    for bx1, by1, bx2, by2 in existing_boxes:
        if not (x2 + margin < bx1 or x1 - margin > bx2 or y2 + margin < by1 or y1 - margin > by2):
            return True
    return False
import cv2
import numpy as np
import random
def paste_card(background, card_img, existing_boxes):
    bg_h, bg_w = background.shape[:2]

    # 초기 스케일 설정
    scale = random.uniform(0.25, 0.45)
    card_h = int(bg_h * scale)
    aspect = card_img.shape[1] / card_img.shape[0]
    card_w = int(card_h * aspect)

    # 크기 조정
    card_img = cv2.resize(card_img, (card_w, card_h))
    h, w = card_img.shape[:2]

    # 랜덤 회전
    angle = random.uniform(-12, 12)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    new_w, new_h = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))

    # 회전 후 배경보다 크면 축소해서 다시 회전
    if new_w >= bg_w or new_h >= bg_h:
        scale_adj = min(bg_w / new_w, bg_h / new_h) * 0.8
        card_img = cv2.resize(card_img, (int(w * scale_adj), int(h * scale_adj)))
        h, w = card_img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        new_w, new_h = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - (w / 2)
    M[1, 2] += (new_h / 2) - (h / 2)
    rotated = cv2.warpAffine(card_img, M, (new_w, new_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0))

    # 마스크 생성
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    mask_bool = mask > 0

    # 위치 결정 (비겹침 보장)
    tries = 0
    while True:
        tries += 1
        if tries > 50:
            logger.warning("No valid placement found, skipping card.")
            return background, np.zeros((bg_h, bg_w), dtype=np.uint8), None

        x_offset = random.randint(0, max(0, bg_w - new_w))
        y_offset = random.randint(0, max(0, bg_h - new_h))
        bbox = [x_offset, y_offset, x_offset + new_w, y_offset + new_h]
        if not check_overlap(bbox, existing_boxes):
            break

    # 배경 유지하며 합성
    composed = background.copy()
    roi = composed[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
    roi[mask_bool] = rotated[mask_bool]
    composed[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = roi

    mask_full = np.zeros((bg_h, bg_w), dtype=np.uint8)
    mask_full[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = mask

    return composed, mask_full, bbox
def save_yolo_label(bbox_list, out_path, img_w, img_h):
    lines = []
    for x1, y1, x2, y2 in bbox_list:
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

# ───────────────────────────────────────────────
def create_split_dirs(split_name):
    split_dir = os.path.join(OUT_DIR, split_name)
    for sub in ["images", "masks", "yolo_labels"]:
        os.makedirs(os.path.join(split_dir, sub), exist_ok=True)
    return split_dir

# ───────────────────────────────────────────────
def create_split_dirs(split_name):
    split_dir = os.path.join(OUT_DIR, split_name)
    for sub in ["images", "masks", "yolo_labels"]:
        os.makedirs(os.path.join(split_dir, sub), exist_ok=True)
    return split_dir

# ───────────────────────────────────────────────
def main(num_images=300):
    backgrounds = load_images(BG_DIR)
    cards = load_images(CARD_DIR)
    logger.info(f"Loaded {len(backgrounds)} backgrounds, {len(cards)} cards.")

    # Split boundaries
    train_cut = int(num_images * SPLITS["train"])
    val_cut = int(num_images * (SPLITS["train"] + SPLITS["val"]))

    for i in tqdm(range(num_images), desc="🎨 Generating dataset"):
        bg_path = random.choice(backgrounds)
        background = cv2.imread(bg_path)
        if background is None:
            continue

        # split 결정
        if i < train_cut:
            split = "train"
        elif i < val_cut:
            split = "val"
        else:
            split = "test"

        split_dir = create_split_dirs(split)
        img_dir = os.path.join(split_dir, "images")
        mask_dir = os.path.join(split_dir, "masks")
        label_dir = os.path.join(split_dir, "yolo_labels")

        bg_h, bg_w = background.shape[:2]
        mask_total = np.zeros((bg_h, bg_w), dtype=np.uint8)
        boxes = []
        composed = background.copy()

        n_cards = random.randint(2, 5)
        selected_cards = random.choices(cards, k=n_cards)

        for card_path in selected_cards:
            card_img = cv2.imread(card_path)
            if card_img is None:
                continue

            composed, mask_card, bbox = paste_card(composed, card_img, boxes)
            if bbox is not None:
                mask_total = cv2.bitwise_or(mask_total, mask_card)
                boxes.append(bbox)

        if not boxes:
            continue

        img_name = f"{split}_{i:03d}.png"
        cv2.imwrite(os.path.join(img_dir, img_name), composed)
        cv2.imwrite(os.path.join(mask_dir, f"{split}_mask_{i:03d}.png"), mask_total)
        save_yolo_label(boxes,
                        os.path.join(label_dir, f"{split}_{i:03d}.txt"),
                        bg_w, bg_h)

        logger.info(f"[{split}] {img_name}: {len(boxes)} cards")

    logger.info("✅ Dataset generation with train/val/test splits completed.")

# ───────────────────────────────────────────────
if __name__ == "__main__":
    main(num_images=300)


