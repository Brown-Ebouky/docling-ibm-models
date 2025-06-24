#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# from PIL import Image, ImageDraw, ImageFont

from docling_ibm_models.layoutmodel.detrex_layout_predictor import DetRexLayoutPredictor
from demo.demo_layout_predictor import save_predictions

def demo(
    logger: logging.Logger,
    artifact_path: str,
    device: str,
    num_threads: int,
    img_dir: str,
    viz_dir: str,
):
    r"""
    Apply LayoutPredictor on the input image directory

    If you want to load from PDF:
    pdf_image = pyvips.Image.new_from_file("test_data/ADS.2007.page_123.pdf", page=0)
    """
    # Create the layout predictor
    lpredictor = DetRexLayoutPredictor(artifact_path, device=device, num_threads=num_threads)

    # Predict all test png images
    t0 = time.perf_counter()
    img_counter = 0
    for img_fn in Path(img_dir).rglob("*.png"):
        img_counter += 1
        logger.info("Predicting '%s'...", img_fn)

        with Image.open(img_fn) as image:
            # Predict layout
            img_t0 = time.perf_counter()
            preds = list(lpredictor.predict(image))
            img_ms = 1000 * (time.perf_counter() - img_t0)
            logger.debug("Prediction(ms): {:.2f}".format(img_ms))

            # Save predictions
            logger.info("Saving prediction visualization in: '%s'", viz_dir)
            save_predictions("ST", viz_dir, img_fn, image, preds)
            if img_counter >= 10:
                break
    total_ms = 1000 * (time.perf_counter() - t0)
    avg_ms = (total_ms / img_counter) if img_counter > 0 else 0
    logger.info(
        "For {} images(ms): [total|avg] = [{:.1f}|{:.1f}]".format(
            img_counter, total_ms, avg_ms
        )
    )


def main(args):
    r""" """
    num_threads = int(args.num_threads) if args.num_threads is not None else None
    device = args.device.lower()
    img_dir = args.img_dir
    viz_dir = args.viz_dir

    # Initialize logger
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("LayoutPredictor")
    logger.setLevel(logging.DEBUG)
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Ensure the viz dir
    Path(viz_dir).mkdir(parents=True, exist_ok=True)

    artifact_path = "bebouky/deterex-custom-model" # os.path.join(download_path, "model_artifacts/layout")

    # Test the LayoutPredictor
    demo(logger, artifact_path, device, num_threads, img_dir, viz_dir)


if __name__ == "__main__":
    r"""
    python -m demo.demo_layout_predictor -i <images_dir>
    """
    parser = argparse.ArgumentParser(description="Test the LayoutPredictor")
    parser.add_argument(
        "-d", "--device", required=False, default="cpu", help="One of [cpu, cuda, mps]"
    )
    parser.add_argument(
        "-n", "--num_threads", required=False, default=4, help="Number of threads"
    )
    parser.add_argument(
        "-i",
        "--img_dir",
        required=True,
        help="PNG images input directory",
    )
    parser.add_argument(
        "-v",
        "--viz_dir",
        required=False,
        default="viz/",
        help="Directory to save prediction visualizations",
    )

    args = parser.parse_args()
    main(args)
