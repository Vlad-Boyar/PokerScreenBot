
# Card crop preprocessing for rank classification (same pipeline as used during model training)

import cv2
import torch

def preprocess_card_crop_opencv(img, img_size=64):
    """
    Preprocess card image crop for rank classification.

    Steps:
    - Adaptive threshold
    - Crop bounding box
    - Resize to square with padding
    - Normalize to [-1, 1] tensor

    Args:
        img: Grayscale OpenCV image
        img_size: Target image size (default 64)

    Returns:
        PyTorch tensor of shape [1, 1, img_size, img_size]
    """
    img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 10)

    coords = cv2.findNonZero(img_bin)
    x, y, w, h = cv2.boundingRect(coords)

    pad = 4
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, img.shape[1])
    y2 = min(y + h + pad, img.shape[0])

    digit_crop = img_bin[y1:y2, x1:x2]

    h_crop, w_crop = digit_crop.shape
    scale = img_size / max(h_crop, w_crop)
    resized = cv2.resize(digit_crop, (int(w_crop * scale), int(h_crop * scale)),
                         interpolation=cv2.INTER_AREA)

    pad_top = (img_size - resized.shape[0]) // 2
    pad_bottom = img_size - resized.shape[0] - pad_top
    pad_left = (img_size - resized.shape[1]) // 2
    pad_right = img_size - resized.shape[1] - pad_left

    digit_square = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                      cv2.BORDER_CONSTANT, value=0)

    tensor = torch.tensor(digit_square, dtype=torch.float32) / 255.0
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    tensor = (tensor - 0.5) / 0.5

    return tensor
