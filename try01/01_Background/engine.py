# -*- coding: utf-8 -*-
"""
engine.py
───────────────────────────────────────────────
명함 + 배경 합성 엔진 (회전 시 원래 배경 유지, 비겹침, 색상 보존)
───────────────────────────────────────────────
"""

import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from loguru import logger
import albumentations as A

# ───────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
CARD_DIR = os.path.join(ROOT_DIR, "00_IMGS/NameCards")
BG_DIR = os.path.join(ROOT_DIR, "00_IMGS/Backgrounds/auto_backgrounds")
OUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
logger.add(os.path.join(LOG_DIR, "augment.log"), level="INFO")

# ───────────────────────────────────────────────
# 회전/기울기 변환: 검정 채움 제거 (알파마스크 활용)
geom_aug = A.Compose([
    A.SafeRotate(limit=12, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), p=1.0),
    A.Perspective(scale=(0.02, 0.05), fit_output=True, pad_val=(0, 0, 0), p=1.0)
])

# ───────────────────────────────────────────────
def load_images(folder):
    imgs = []
    if not os.path.exists(folder):
        return imgs
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(folder, fn)
            img = cv2.imread(path)
            if img is not None:
                imgs.append(img)
    return imgs

# ───────────────────────────────────────────────
def resize_and_crop_center(img, target_size=(1920, 1080)):
    h, w, _ = img.shape
    tw, th = target_size
    scale = max(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))
    x0 = (nw - tw) // 2
    y0 = (nh - th) // 2
    return resized[y0:y0 + th, x0:x0 + tw]

# ───────────────────────────────────────────────
def check_overlap(x, y, w, h, placed_boxes):
    for (px, py, pw, ph) in placed_boxes:
        if not (x + w < px or px + pw < x or y + h < py or py + ph < y):
            return True
    return False

# ───────────────────────────────────────────────
def random_paste(bg, cards, n_cards):
    """명함 회전 후 생긴 여백은 투명처리하고 배경 유지"""
    canvas = bg.copy()
    h_bg, w_bg, _ = bg.shape
    placed_boxes = []

    for _ in range(n_cards):
        card = random.choice(cards)

        # 크기 조정 (배경 폭 대비 25~35%)
        scale = random.uniform(0.25, 0.35)
        target_w = int(w_bg * scale)
        ratio = target_w / card.shape[1]
        target_h = int(card.shape[0] * ratio)
        card = cv2.resize(card, (target_w, target_h))

        # 회전 + 기울이기
        aug_card = geom_aug(image=card)["image"]

        # 검정 여백 부분을 mask로 추출
        gray = cv2.cvtColor(aug_card, cv2.COLOR_BGR2GRAY)
        mask = (gray > 15).astype(np.uint8) * 255  # 명함 영역만 255
        bbox = cv2.boundingRect(mask)
        x0, y0, w, h = bbox
        aug_card = aug_card[y0:y0 + h, x0:x0 + w]
        mask = mask[y0:y0 + h, x0:x0 + w]

        # 배경에 비겹침 위치 선정
        for _ in range(50):
            x = random.randint(0, max(1, w_bg - w))
            y = random.randint(0, max(1, h_bg - h))
            if not check_overlap(x, y, w, h, placed_boxes):
                placed_boxes.append((x, y, w, h))
                break

        # 명함 영역만 배경에 합성 (여백은 투명)
        roi = canvas[y:y + h, x:x + w]
        mask3 = cv2.merge([mask, mask, mask])
        inv_mask = 255 - mask3
        blended = cv2.add(cv2.bitwise_and(roi, inv_mask), cv2.bitwise_and(aug_card, mask3))
        canvas[y:y + h, x:x + w] = blended

    return canvas

# ───────────────────────────────────────────────
def main(num_images=50):
    cards = load_images(CARD_DIR)
    bgs = load_images(BG_DIR)
    if not cards:
        logger.error(f"❌ No cards found in {CARD_DIR}")
        return
    if not bgs:
        logger.error(f"❌ No backgrounds found in {BG_DIR}")
        return

    logger.info(f"Loaded {len(cards)} cards and {len(bgs)} backgrounds")

    for i in tqdm(range(num_images), desc="🧩 Generating Augmented Images"):
        bg = resize_and_crop_center(random.choice(bgs), (1920, 1080))
        n_cards = random.randint(2, min(4, len(cards)))
        result = random_paste(bg, cards, n_cards)
        out_path = os.path.join(OUT_DIR, f"aug_{i:03d}.png")
        cv2.imwrite(out_path, result)
        logger.info(f"[{i+1}/{num_images}] {out_path} | cards={n_cards}")

    logger.info("✅ All augmentations complete.")

# ───────────────────────────────────────────────
if __name__ == "__main__":
    main(num_images=100)
