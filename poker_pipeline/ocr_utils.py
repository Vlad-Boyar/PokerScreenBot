
"""
OCR utilities for digit extraction from image crops.

Uses YOLO model to detect digits and reconstruct numerical values.
"""

import numpy as np
from ultralytics import YOLO
import logging
from poker_pipeline.models_loader import YOLO_DIGITS_PATH

yolo_digits = YOLO(YOLO_DIGITS_PATH)

digit_class_map = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "."]

def extract_number_from_crop(crop_img):
    """
    Extract numerical value from an image crop using YOLO-based digit detector.

    Args:
        crop_img: OpenCV image (BGR or grayscale) containing digits.

    Returns:
        Float value extracted from the image.
        If extraction fails, returns 0.0.
    """
    res_digits = yolo_digits.predict(source=crop_img, imgsz=416, conf=0.2, iou=0.5, device=0, verbose=False)[0]

    digit_boxes = res_digits.boxes.xyxy.cpu().numpy()
    digit_classes = res_digits.boxes.cls.cpu().numpy().astype(int)

    if len(digit_boxes) == 0:
        return 0.0

    x_centers = (digit_boxes[:, 0] + digit_boxes[:, 2]) / 2
    sorted_indices = np.argsort(x_centers)

    number_str = ""
    for idx in sorted_indices:
        cls = digit_classes[idx]
        number_str += digit_class_map[cls]

    try:
        return float(number_str)
    except:
        logging.debug(f"⚠️ Failed to parse number: '{number_str}'")
        return 0.0
