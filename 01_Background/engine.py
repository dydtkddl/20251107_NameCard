# -*- coding: utf-8 -*-
"""
engine.py
───────────────────────────────────────────────
명함 + 배경 합성 자동 증강 엔진 (3D 회전, 비겹침, 색상 보존, 흐림 제거)
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
ROOT_DIR = os.path.dirname(BASE_DIR)
CARD_DIR = os.path.join(ROOT_DIR, "00_IMGS/NameCards")
BG_DIR = os.path.join(ROOT_DIR, "00_IMGS/Backgrounds/auto_backgrounds")
OUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
logger.add(os.path.join(LOG_DIR, "augment.log"), level="INFO")

# ───────────────────────────────────────────────
# Albumentations 변환 (색상 변경 없이 기하학 변형만)
geom_aug = A.Compose([
    A.SafeRotate(limit=12, border_mode=cv2.BORDER_CONSTANT, p=1.0),  # value 제거 (경고 방지)
    A.Perspective(scale=(0.02, 0.05), fit_output=True, p=1.0)
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
def resize_and_crop_center(img, target_size=(1920, 1080)):
    """배경 이미지를 1920×1080으로 중앙 기준 crop"""
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
    """명함끼리 겹침 여부 검사"""
    for (px, py, pw, ph) in placed_boxes:
        if not (x + w < px or px + pw < x or y + h < py or py + ph < y):
            return True
    return False

# ───────────────────────────────────────────────
def random_paste(bg, cards, n_cards):
    """배경 위에 명함을 자연스럽게 배치 (비겹침 + 흐림 없음)"""
    canvas = bg.copy()
    h_bg, w_bg, _ = bg.shape
    placed_boxes = []

    for _ in range(n_cards):
        card = random.choice(cards)

        # 명함 크기: 배경 폭 대비 25~35%
        scale = random.uniform(0.25, 0.35)
        target_w = int(w_bg * scale)
        ratio = target_w / card.shape[1]
        target_h = int(card.shape[0] * ratio)
        card = cv2.resize(card, (target_w, target_h))

        # 3D 회전만 적용 (색상 유지)
        card = geom_aug(image=card)["image"]

        # 겹치지 않게 랜덤 위치 선정
        for _ in range(50):
            x = random.randint(0, max(1, w_bg - card.shape[1]))
            y = random.randint(0, max(1, h_bg - card.shape[0]))
            if not check_overlap(x, y, card.shape[1], card.shape[0], placed_boxes):
                placed_boxes.append((x, y, card.shape[1], card.shape[0]))
                break

        # 단순 덮어쓰기 (blur 없음)
        canvas[y:y + card.shape[0], x:x + card.shape[1]] = card

    return canvas

# ───────────────────────────────────────────────
def main(num_images=50):
    """엔진 메인 루프"""
    cards = load_images(CARD_DIR)
    bgs = load_images(BG_DIR)

    if not cards:
        logger.error(f"❌ No cards found in {CARD_DIR}")
        return
    if not bgs:
        logger.error(f"❌ No backgrounds found in {BG_DIR}")
        return

    logger.info(f"Loaded {len(cards)} cards and {len(bgs)} backgrounds.")

    for i in tqdm(range(num_images), desc="🧩 Generating Augmented Images"):
        bg = resize_and_crop_center(random.choice(bgs), (1920, 1080))
        n_cards = random.randint(2, min(4, len(cards)))  # 2~4장
        result = random_paste(bg, cards, n_cards)
        out_path = os.path.join(OUT_DIR, f"aug_{i:03d}.png")
        cv2.imwrite(out_path, result)
        logger.info(f"[{i+1}/{num_images}] {out_path} | cards={n_cards}")

    logger.info("✅ All augmentations complete.")

# ───────────────────────────────────────────────
if __name__ == "__main__":
    main(num_images=100)

