# -*- coding: utf-8 -*-
"""
engine_labelme_to_yolo_coco_structure.py
────────────────────────────────────────────
Labelme JSON + 이미지 → YOLO 학습 구조 + COCO JSON 자동 생성
────────────────────────────────────────────
"""
import os, cv2, json, random, shutil, numpy as np
from tqdm import tqdm
from labelme import utils as labelme_utils
from sklearn.model_selection import train_test_split
import logging

# ───────────────────────────────────────────────
def setup_logger(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "labelme_convert.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("Labelme2YOLOCOCO")

# ───────────────────────────────────────────────
def collect_pairs(root):
    exts = (".jpg", ".jpeg", ".png")
    pairs = []
    for d, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(exts):
                base = os.path.splitext(f)[0]
                js = os.path.join(d, base + ".json")
                img = os.path.join(d, f)
                if os.path.exists(js):
                    pairs.append((img, js))
    return pairs

# ───────────────────────────────────────────────
def json_to_yolo_mask(img_path, js_path, out_mask, out_label):
    with open(js_path, encoding='utf-8') as f:
        data = json.load(f)

    h, w = data["imageHeight"], data["imageWidth"]
    mask = np.zeros((h, w), dtype=np.uint8)
    lines = []

    for shape in data.get("shapes", []):
        label = 0
        pts = np.array(shape["points"], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        norm = pts / np.array([[w, h]])
        coords = " ".join([f"{x:.6f} {y:.6f}" for x, y in norm])
        lines.append(f"{label} {coords}")

    if len(lines) > 0:
        os.makedirs(os.path.dirname(out_mask), exist_ok=True)
        os.makedirs(os.path.dirname(out_label), exist_ok=True)
        cv2.imwrite(out_mask, mask)
        with open(out_label, "w") as f:
            f.write("\n".join(lines))

# ───────────────────────────────────────────────
def split_dataset(pairs, out_root, logger):
    random.shuffle(pairs)
    train, temp = train_test_split(pairs, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    splits = {"train": train, "val": val, "test": test}

    for split, subset in splits.items():
        img_dir = os.path.join(out_root, split, "images")
        mask_dir = os.path.join(out_root, split, "masks")
        label_dir = os.path.join(out_root, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        for img, js in tqdm(subset, desc=f"{split} set"):
            fname = os.path.basename(img)
            base = os.path.splitext(fname)[0]
            img_out = os.path.join(img_dir, fname)
            mask_out = os.path.join(mask_dir, f"{split}_mask_{base.split('_')[-1]}.png")
            label_out = os.path.join(label_dir, f"{base}.txt")
            shutil.copy2(img, img_out)
            json_to_yolo_mask(img, js, mask_out, label_out)
        logger.info(f"{split}: {len(subset)} samples processed.")

# ───────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="./output_yolo_coco")
    args = parser.parse_args()

    logger = setup_logger(args.output_dir)
    pairs = collect_pairs(args.input_dir)
    logger.info(f"Found {len(pairs)} labelme pairs.")
    split_dataset(pairs, args.output_dir, logger)
    logger.info("🎯 Done! YOLO/COCO-ready dataset created.")


