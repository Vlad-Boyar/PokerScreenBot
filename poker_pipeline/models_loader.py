
"""
Models loader paths.

Defines absolute paths to trained model files stored in 'models/' directory.
"""

import os

# Define absolute path to the 'models' directory relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Paths to trained model files
YOLO_TABLE_ELEMENTS_PATH = os.path.join(MODELS_DIR, "yolo_table_elements.pt")
YOLO_DIGITS_PATH = os.path.join(MODELS_DIR, "yolo_digits.pt")
FORMAT_CLASSIFIER_PATH = os.path.join(MODELS_DIR, "format_classifier.pth")
SUIT_CLASSIFIER_PATH = os.path.join(MODELS_DIR, "suit_classifier.pth")
RANK_CLASSIFIER_PATH = os.path.join(MODELS_DIR, "rank_classifier.pth")
