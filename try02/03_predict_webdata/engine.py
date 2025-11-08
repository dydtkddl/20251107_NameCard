# -*- coding: utf-8 -*-
"""
predict_card.py
──────────────────────────────────────────────
YOLOv8-Seg Inference Script with Config.json

Features:
- Uses external config.json (default: ./config.json)
- Auto logging (file + console)
- tqdm progress bar
- Robust cv2-based image validation (skips broken/unreadable files)
──────────────────────────────────────────────
"""

import os
import json
import cv2
import argparse
import logging
from tqdm import tqdm
from ultralytics import YOLO


# ───────────────────────────────────────────────
def load_config(config_path: str):
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[X] config.json not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    logging.info(f"✅ Config loaded from: {config_path}")
    return cfg


# ───────────────────────────────────────────────
def setup_logger(output_dir: str):
    """Set up logging to both file and console."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "predict.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    logging.info("🚀 YOLOv8-Seg Prediction Started")
    return log_file


# ───────────────────────────────────────────────
def main(config_path: str):
    """Main inference pipeline."""
    # 1️⃣ Load Config
    cfg = load_config(config_path)

    model_path = os.path.abspath(cfg["model_path"])
    source_dir = os.path.abspath(cfg["source_dir"])
    output_dir = os.path.abspath(cfg.get("output_dir", os.path.join(os.getcwd(), "runs")))
    run_name = cfg.get("run_name", "predictions_webdata")

    # 2️⃣ Logging setup
    setup_logger(output_dir)

    # 3️⃣ Load model
    logging.info(f"🧠 Loading YOLOv8 model from: {model_path}")
    model = YOLO(model_path)
    logging.info("✅ Model loaded successfully")

    # 4️⃣ Gather valid images
    valid_ext = (".jpg", ".jpeg", ".png")
    all_files = os.listdir(source_dir)
    valid_images = [f for f in all_files if f.lower().endswith(valid_ext)]

    if not valid_images:
        logging.error(f"[X] No valid image files found in: {source_dir}")
        return

    logging.info(f"📂 Found {len(valid_images)} image(s) in: {source_dir}")

    # 5️⃣ Loop through images with tqdm
    for fn in tqdm(valid_images, desc="🔍 Running predictions", ncols=100):
        img_path = os.path.join(source_dir, fn)
        img = cv2.imread(img_path)

        if img is None:
            logging.warning(f"⚠️ Skipping unreadable image: {fn}")
            continue

        try:
            results = model.predict(
                source=img_path,
                conf=cfg.get("conf", 0.5),
                iou=cfg.get("iou", 0.5),
                imgsz=cfg.get("imgsz", 640),
                save=True,
                save_txt=True,
                save_conf=True,
                show_boxes=True,
                show_conf=True,
                show_labels=True,
                project=output_dir,
                name=run_name,
                device=cfg.get("device", 0),
                verbose=False
            )

            # Extract save_dir from result object
            save_dir = None
            if isinstance(results, list) and len(results) > 0 and hasattr(results[0], "save_dir"):
                save_dir = results[0].save_dir
            elif hasattr(results, "save_dir"):
                save_dir = results.save_dir

            if save_dir:
                logging.info(f"✅ Done: {fn} → Saved to {save_dir}")
            else:
                logging.warning(f"⚠️ No output path found for {fn}")

        except Exception as e:
            logging.error(f"❌ Error while predicting {fn}: {e}")

    logging.info("🏁 All predictions completed successfully.")


# ───────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8-Seg Predict Script with Config JSON")
    parser.add_argument("--config", type=str, default="./config.json",
                        help="Path to config.json (default: ./config.json)")
    args = parser.parse_args()

    main(args.config)



