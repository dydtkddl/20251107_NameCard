# -*- coding: utf-8 -*-
"""
engine.py
───────────────────────────────────────────────
명함 + 배경 합성 자동 증강 엔진
───────────────────────────────────────────────
실행:  python engine.py
출력:  output/aug_###.png
"""

import os
import cv2
import random
import numpy as np
from tqdm import tqdm
from loguru import logger
import albumentations as A

# ───────────────────────────────────────────────
# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))             # 01_Background/
CARD_DIR = os.path.join(BASE_DIR, "../00_NameCard")               # ../00_NameCard/
BG_DIR = BASE_DIR                                                 # 현재 폴더 자체에 배경 있음
OUT_DIR = os.path.join(BASE_DIR, "output")                        # ./output/
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
logger.add(os.path.join(LOG_DIR, "augment.log"), level="INFO")

# ───────────────────────────────────────────────
# 증강 파이프라인 (기하학적 + 광학적)
geom_aug = A.Compose([
    A.Rotate(limit=25, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255), p=0.9),
    A.Perspective(scale=(0.05, 0.15), p=0.8),
    A.RandomBrightnessContrast(p=0.6),
    A.MotionBlur(blur_limit=5, p=0.3),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
])

# ───────────────────────────────────────────────
def load_images(folder):
    """폴더 내 모든 이미지 불러오기"""
    imgs = []
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(folder, fn)
            img = cv2.imread(path)
            if img is not None:
                imgs.append(img)
    return imgs

# ───────────────────────────────────────────────
def random_paste(bg, cards, n_cards):
    """배경에 명함 n장을 랜덤 배치"""
    canvas = bg.copy()
    h_bg, w_bg, _ = bg.shape

    for _ in range(n_cards):
        card = random.choice(cards)
        scale = random.uniform(0.4, 0.8)
        card = cv2.resize(card, (int(card.shape[1]*scale), int(card.shape[0]*scale)))

        # Albumentations 증강 적용
        aug = geom_aug(image=card)
        card = aug["image"]

        # 랜덤 배치 좌표
        x = random.randint(0, max(1, w_bg - card.shape[1]))
        y = random.randint(0, max(1, h_bg - card.shape[0]))

        # 자연스러운 seamless 합성
        mask = 255 * np.ones(card.shape, card.dtype)
        center = (x + card.shape[1] // 2, y + card.shape[0] // 2)
        try:
            canvas = cv2.seamlessClone(card, canvas, mask[:, :, 0], center, cv2.NORMAL_CLONE)
        except:
            # 경계 벗어나면 그냥 paste
            canvas[y:y+card.shape[0], x:x+card.shape[1]] = card

    return canvas

# ───────────────────────────────────────────────
def main(num_images=100, output_size=(1280, 720)):
    cards = load_images(CARD_DIR)
    bgs = load_images(BG_DIR)

    if not cards:
        logger.error(f"No cards found in {CARD_DIR}")
        return
    if not bgs:
        logger.error(f"No backgrounds found in {BG_DIR}")
        return

    logger.info(f"Loaded {len(cards)} cards and {len(bgs)} backgrounds")

    for i in tqdm(range(num_images), desc="🧩 Generating Augmented Images"):
        bg = random.choice(bgs)
        bg = cv2.resize(bg, output_size)

        n_cards = random.randint(1, min(4, len(cards)))  # 명함 수 1~4 랜덤
        result = random_paste(bg, cards, n_cards)

        out_path = os.path.join(OUT_DIR, f"aug_{i:03d}.png")
        cv2.imwrite(out_path, result)
        logger.info(f"[{i+1}/{num_images}] → {out_path} | cards={n_cards}")

    logger.info("✅ All augmentations complete.")

# ───────────────────────────────────────────────
if __name__ == "__main__":
    main(num_images=100)

