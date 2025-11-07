# -*- coding: utf-8 -*-
"""
engine.py
───────────────────────────────────────────────
명함 + 배경 합성 자동 증강 엔진 (ver. for /01_Background)
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
# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # /01_Background
ROOT_DIR = os.path.dirname(BASE_DIR)                   # /20251107_NameCard
CARD_DIR = os.path.join(ROOT_DIR, "00_IMGS/NameCards")
BG_DIR = os.path.join(ROOT_DIR, "00_IMGS/Backgrounds/auto_backgrounds")
OUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
logger.add(os.path.join(LOG_DIR, "augment.log"), level="INFO")

# ───────────────────────────────────────────────
# Albumentations 증강 파이프라인
geom_aug = A.Compose([
    A.SafeRotate(limit=25, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255), p=0.9),
    A.Perspective(scale=(0.05, 0.15), p=0.8),
    A.RandomBrightnessContrast(p=0.6),
    A.MotionBlur(blur_limit=5, p=0.3),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
])

# ───────────────────────────────────────────────
def load_images(folder):
    """폴더 내 모든 이미지 로드"""
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
def random_paste(bg, cards, n_cards):
    """배경 위에 n장의 명함 랜덤 배치"""
    canvas = bg.copy()
    h_bg, w_bg, _ = bg.shape

    for _ in range(n_cards):
        card = random.choice(cards)

        # 크기 조절
        scale = random.uniform(0.4, 0.8)
        card = cv2.resize(card, (int(card.shape[1] * scale), int(card.shape[0] * scale)))

        # 기하학적 증강
        card = geom_aug(image=card)["image"]

        # 랜덤 위치 배치
        x = random.randint(0, max(1, w_bg - card.shape[1]))
        y = random.randint(0, max(1, h_bg - card.shape[0]))

        # 자연스러운 합성
        mask = 255 * np.ones(card.shape, card.dtype)
        center = (x + card.shape[1] // 2, y + card.shape[0] // 2)
        try:
            canvas = cv2.seamlessClone(card, canvas, mask[:, :, 0], center, cv2.NORMAL_CLONE)
        except Exception:
            # 경계 초과 시 단순 덮어쓰기
            canvas[y:y + card.shape[0], x:x + card.shape[1]] = card

    return canvas

# ───────────────────────────────────────────────
def main(num_images=100, output_size=(1280, 720)):
    # 이미지 로드
    cards = load_images(CARD_DIR)
    bgs = load_images(BG_DIR)

    if not cards:
        logger.error(f"❌ No cards found in {CARD_DIR}")
        return
    if not bgs:
        logger.error(f"❌ No backgrounds found in {BG_DIR}")
        logger.error("👉 먼저 background_downloader.py를 실행해 배경 이미지를 확보하세요.")
        return

    logger.info(f"Loaded {len(cards)} cards and {len(bgs)} backgrounds.")

    # 합성 루프
    for i in tqdm(range(num_images), desc="🧩 Generating Augmented Images"):
        bg = random.choice(bgs)
        bg = cv2.resize(bg, output_size)

        n_cards = random.randint(1, min(4, len(cards)))
        result = random_paste(bg, cards, n_cards)

        out_path = os.path.join(OUT_DIR, f"aug_{i:03d}.png")
        cv2.imwrite(out_path, result)
        logger.info(f"[{i+1}/{num_images}] → {out_path} | cards={n_cards}")

    logger.info("✅ All augmentations complete.")

# ───────────────────────────────────────────────
if __name__ == "__main__":
    main(num_images=50)

