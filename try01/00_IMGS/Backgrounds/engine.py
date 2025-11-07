# -*- coding: utf-8 -*-
"""
background_downloader.py
───────────────────────────────
Unsplash에서 배경 이미지를 자동 다운로드
───────────────────────────────
"""

import os
import requests
from tqdm import tqdm
from loguru import logger

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auto_backgrounds")
os.makedirs(SAVE_DIR, exist_ok=True)
logger.add(os.path.join(SAVE_DIR, "download.log"), level="INFO")

KEYWORDS = [
    "wood desk texture",
    "forest floor background",
    "concrete floor texture",
    "paper background",
    "fabric texture"
]


def get_image_urls(keyword, count=10):
    ACCESS_KEY = "OzfKTM_mlsPjnCGETeh16ixjYH5HC2I3XTvqeP8mYLQ"
    api_url = f"https://api.unsplash.com/search/photos?query={keyword}&per_page={count}&client_id={ACCESS_KEY}"

    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(api_url, headers=headers)
    if resp.status_code != 200:
        logger.error(f"[{keyword}] 요청 실패: {resp.status_code}")
        return []
    data = resp.json()
    return [d["urls"]["regular"] for d in data["results"]]


def download_backgrounds():
    total = 0
    for kw in tqdm(KEYWORDS, desc="🌄 Downloading backgrounds"):
        urls = get_image_urls(kw, count=10)
        for url in urls:
            fname = os.path.join(SAVE_DIR, os.path.basename(url.split("?")[0]) + ".jpg")
            if os.path.exists(fname):
                continue
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    with open(fname, "wb") as f:
                        f.write(r.content)
                    total += 1
            except Exception as e:
                logger.error(f"❌ {url} 다운로드 실패: {e}")
    logger.info(f"✅ 총 {total}장의 배경 이미지 다운로드 완료.")


if __name__ == "__main__":
    download_backgrounds()

