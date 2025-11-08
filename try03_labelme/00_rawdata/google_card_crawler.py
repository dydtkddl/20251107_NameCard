# -*- coding: utf-8 -*-
"""
google_card_crawler.py
────────────────────────────────────────────
Google 이미지에서 한국 명함 이미지를 자동 크롤링하는 엔진

사용 예시:
    python google_card_crawler.py --query "한국 명함" --limit 1000
"""

import os
import logging
import argparse
from tqdm import tqdm
from icrawler.builtin import GoogleImageCrawler
from PIL import Image

# ───────────────────────────────────────────────
def setup_logger(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler())

# ───────────────────────────────────────────────
def remove_invalid_images(folder: str):
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
    all_files = [f for f in os.listdir(folder) if f.lower().endswith(valid_ext)]

    for f in tqdm(all_files, desc="🧹 Checking images", ncols=100):
        path = os.path.join(folder, f)
        try:
            img = Image.open(path)
            img.verify()
        except Exception as e:
            logging.warning(f"❌ Removing invalid image: {f} ({e})")
            os.remove(path)

from icrawler import downloader

class MyGoogleImageCrawler(GoogleImageCrawler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        })

# ───────────────────────────────────────────────
def crawl_google_images(query: str, limit: int, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    crawler = MyGoogleImageCrawler(storage={"root_dir": output_dir})

    crawler.crawl(
        keyword=query,
        max_num=limit,
        min_size=(200, 200),
        max_size=None,
        file_idx_offset=0
    )

# ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Google Image Crawler for Korean Business Cards")
    parser.add_argument("--query", type=str, default="한국 명함", help="검색할 키워드")
    parser.add_argument("--limit", type=int, default=1000, help="다운로드할 이미지 개수")
    parser.add_argument("--output", type=str, default="./data_raw", help="출력 폴더 경로")
    args = parser.parse_args()

    log_path = os.path.join(args.output, "crawl.log")
    setup_logger(log_path)

    logging.info(f"🚀 Start crawling query='{args.query}', limit={args.limit}")
    crawl_google_images(args.query, args.limit, args.output)
    logging.info("✅ Crawling finished. Start validation...")

    remove_invalid_images(args.output)
    logging.info("🏁 All done! Valid images saved in " + args.output)

# ───────────────────────────────────────────────
if __name__ == "__main__":
    main()


