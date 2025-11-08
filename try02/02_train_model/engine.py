# -*- coding: utf-8 -*-
"""
engine_train_yolov8seg.py
───────────────────────────────────────────────
YOLOv8-Seg 기반 명함 객체 탐지 fine-tuning 전체 파이프라인
───────────────────────────────────────────────
"""

import os
import sys
import time
import random
import logging
import numpy as np
import torch
from datetime import datetime
from ultralytics import YOLO
from tqdm import tqdm
from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd

# ───────────────────────────────────────────────
# 기본 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATASET_DIR = os.path.join(ROOT_DIR, "01_makedataset", "output")

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

LOG_DIR = os.path.join(BASE_DIR, "logs")
RUN_DIR = os.path.join(BASE_DIR, "runs")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RUN_DIR, exist_ok=True)

# ───────────────────────────────────────────────
# 재현성 확보를 위한 시드 고정
def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"🔒 Random seed fixed to {seed}")

fix_seed(42)

# ───────────────────────────────────────────────
# Logging 설정
time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"train_yolov8seg_{time_tag}.log")

logger.remove()
logger.add(sys.stdout, colorize=True,
           format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                  "<level>{level: <8}</level> | <cyan>{message}</cyan>")
logger.add(LOG_FILE, level="DEBUG",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

# Ultralytics 내부 로깅도 기록
ultra_logger = logging.getLogger("ultralytics")
ultra_logger.setLevel(logging.INFO)
ultra_handler = logging.FileHandler(LOG_FILE)
ultra_handler.setFormatter(logging.Formatter("%(asctime)s | [YOLO] %(message)s"))
ultra_logger.addHandler(ultra_handler)

# ───────────────────────────────────────────────
# data.yaml 자동 생성
DATA_YAML = os.path.join(BASE_DIR, "data.yaml")
with open(DATA_YAML, "w") as f:
    f.write(f"""# YOLOv8-Seg dataset config
path: {DATASET_DIR}
train: train/images
val: val/images
test: test/images
nc: 1
names: ["card"]
""")
logger.info(f"🧾 data.yaml created → {DATA_YAML}")

# ───────────────────────────────────────────────
# 모델 로드
MODEL_PATH = os.path.join(BASE_DIR, "yolov8x-seg.pt")
if not os.path.exists(MODEL_PATH):
    logger.error(f"❌ Model checkpoint not found: {MODEL_PATH}")
    sys.exit(1)

logger.info(f"✅ Loaded pretrained YOLOv8 model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# ───────────────────────────────────────────────
def train_yolo(epochs=50, batch=8, imgsz=640, lr=1e-4):
    logger.info("🚀 Starting YOLOv8-Seg training...")
    start_time = time.time()

    try:
        results = model.train(
            data=DATA_YAML,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr,
            optimizer="AdamW",
            device=0,
            project=RUN_DIR,
            name=f"card_seg_train_{time_tag}",
            pretrained=True,
            save=True,
            exist_ok=True,
            verbose=True
        )
    except Exception as e:
        logger.exception(f"❌ Training failed: {e}")
        sys.exit(1)

    elapsed = time.time() - start_time
    logger.success(f"✅ Training completed in {elapsed/60:.2f} min")
    logger.info(f"📁 Results saved at: {results.save_dir}")

    # 결과 CSV 파일 저장 확인
    results_csv = os.path.join(results.save_dir, "results.csv")
    if os.path.exists(results_csv):
        logger.info(f"📊 Training log CSV found → {results_csv}")
    else:
        logger.warning("⚠️ results.csv not found (YOLO internal issue?)")

    return results.save_dir

# ───────────────────────────────────────────────
def validate_yolo():
    logger.info("🔎 Starting validation on val dataset...")
    start_time = time.time()

    try:
        results = model.val(data=DATA_YAML, imgsz=640, device=0)
    except Exception as e:
        logger.exception(f"❌ Validation failed: {e}")
        return None

    elapsed = time.time() - start_time
    logger.success(f"✅ Validation done in {elapsed:.2f}s | "
                   f"mAP50={results.box.map50:.4f}, mAP50-95={results.box.map:.4f}")
    return results

# ───────────────────────────────────────────────
def test_yolo():
    logger.info("🧪 Starting inference on test set...")
    out_dir = os.path.join(RUN_DIR, f"predictions_{time_tag}")
    os.makedirs(out_dir, exist_ok=True)

    image_dir = os.path.join(TEST_DIR, "images")
    test_images = [f for f in os.listdir(image_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    for fn in tqdm(test_images, desc="🔍 Inference", ncols=100):
        img_path = os.path.join(image_dir, fn)
        model.predict(
            source=img_path,
            imgsz=640,
            conf=0.4,
            iou=0.5,
            save=True,
            save_txt=True,
            save_conf=True,
            project=out_dir,
            name="cards",
            device=0,
            verbose=False
        )

    logger.success(f"✅ Inference complete for {len(test_images)} images.")
    logger.info(f"📦 Predictions saved to: {out_dir}")

# ───────────────────────────────────────────────
def summarize_logs(results_dir):
    """결과 요약 및 학습 곡선 시각화"""
    results_csv = os.path.join(results_dir, "results.csv")
    if not os.path.exists(results_csv):
        logger.warning("⚠️ results.csv not found for summary.")
        return

    df = pd.read_csv(results_csv)
    summary_txt = os.path.join(LOG_DIR, f"summary_{time_tag}.txt")
    with open(summary_txt, "w") as s:
        s.write(df.tail(10).to_string(index=False))
    logger.info(f"🧾 Summary extracted → {summary_txt}")

    # Plot 학습 곡선
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df["epoch"], df["train/cls_loss"], label="train_cls_loss", alpha=0.7)
    ax1.plot(df["epoch"], df["val/cls_loss"], label="val_cls_loss", alpha=0.7)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.grid(True)

    ax2 = ax1.twinx()
    if "metrics/mAP50(B)" in df.columns:
        ax2.plot(df["epoch"], df["metrics/mAP50(B)"], "g--", label="mAP50(Box)")
    if "metrics/mAP50(M)" in df.columns:
        ax2.plot(df["epoch"], df["metrics/mAP50(M)"], "r--", label="mAP50(Mask)")
    ax2.set_ylabel("mAP")
    ax2.legend(loc="lower right")

    plt.title("YOLOv8-Seg Training Summary")
    plt.tight_layout()
    fig_path = os.path.join(results_dir, f"training_curve_{time_tag}.png")
    plt.savefig(fig_path)
    logger.info(f"📈 Saved training curve → {fig_path}")

# ───────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=== [YOLOv8-Seg Fine-tuning Pipeline Start] ===")
    save_dir = train_yolo(epochs=50, batch=8)
    validate_yolo()
    test_yolo()
    summarize_logs(save_dir)
    logger.info("=== [Pipeline Finished Successfully] ===")





