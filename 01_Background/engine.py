# -*- coding: utf-8 -*-
"""
engine.py
───────────────────────────────────────────────
명함 + 배경 합성 증강 엔진 (카메라 해상도 기반)
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
# 증강: 색조는 거의 유지, 기하학만 적용
geom_aug = A.Compose([
    A.SafeRotate(limit=10, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255), p=0.9),
    A.Perspective(scale=(0.02, 0.07), p=0.7),
    A.MotionBlur(blur_limit=3, p=0.2),
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
def center_crop_and_resize(img, size=(1920, 1080)):
    """배경을 지정 크기로 맞추되, 비율 유지하며 초과는 중앙 자르기"""
    h, w, _ = img.shape
    target_w, target_h = size

    # 비율 유지하면서 최소한 target 이상 되도록 스케일
    scale = max(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))

    # 중앙 crop
    x_start = (new_w - target_w) // 2
    y_start = (new_h - target_h) // 2
    cropped = resized[y_start:y_start + target_h, x_start:x_start + target_w]
    return cropped

# ───────────────────────────────────────────────
def random_paste(bg, cards, n_cards):
    """배경 위에 명함을 자연스럽게 배치"""
    canvas = bg.copy()
    h_bg, w_bg, _ = bg.shape

    for _ in range(n_cards):
        card = random.choice(cards)

        # 명함 크기 조절 (배경 대비 1/4~1/3)
        scale = random.uniform(0.25, 0.35)
        target_w = int(w_bg * scale)
        ratio = target_w / card.shape[1]
        target_h = int(card.shape[0] * ratio)
        card = cv2.resize(card, (target_w, target_h))

        # 기하학적 증강만 적용 (색조X)
        card = geom_aug(image=card)["image"]

        # 랜덤 위치
        x = random.randint(0, max(1, w_bg - card.shape[1]))
        y = random.randint(0, max(1, h_bg - card.shape[0]))

        # 합성
        mask = 255 * np.ones(card.shape, card.dtype)
        center = (x + card.shape[1] // 2, y + card.shape[0] // 2)
        try:
            canvas = cv2.seamlessClone(card, canvas, mask[:, :, 0], center, cv2.NORMAL_CLONE)
        except Exception:
            canvas[y:y + card.shape[0], x:x + card.shape[1]] = card

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
        bg = random.choice(bgs)
        bg = center_crop_and_resize(bg, (1920, 1080))

        n_cards = random.randint(1, min(3, len(cards)))
        result = random_paste(bg, cards, n_cards)

        out_path = os.path.join(OUT_DIR, f"aug_{i:03d}.png")
        cv2.imwrite(out_path, result)
        logger.info(f"[{i+1}/{num_images}] {out_path} | cards={n_cards}")

    logger.info("✅ All augmentations complete.")

# ───────────────────────────────────────────────
if __name__ == "__main__":
    main(num_images=100)

