"""
Poker table processing pipeline.

Contains model loading and full process_screen pipeline to parse poker table screenshots.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import models
from ultralytics import YOLO
import torchvision.transforms as transforms
from PIL import Image
import math
import logging
from poker_pipeline.cards_preprocessing import preprocess_card_crop_opencv
from poker_pipeline.ocr_utils import extract_number_from_crop
from poker_pipeline.card_classification import (classify_rank, classify_suit)

from poker_pipeline.models_loader import (
    YOLO_TABLE_ELEMENTS_PATH, 
    YOLO_DIGITS_PATH, 
    FORMAT_CLASSIFIER_PATH, 
    SUIT_CLASSIFIER_PATH, 
    RANK_CLASSIFIER_PATH
)

# === Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === YOLO models
yolo_boxes = YOLO(YOLO_TABLE_ELEMENTS_PATH)
yolo_digits = YOLO(YOLO_DIGITS_PATH)

# === ResNet18 model → class_format
class_format = resnet18(weights=None)
num_ftrs = class_format.fc.in_features
class_format.fc = nn.Linear(num_ftrs, 4)
class_format = class_format.to(device)
class_format.load_state_dict(torch.load(FORMAT_CLASSIFIER_PATH, map_location=device))
class_format.eval()

# === MobileNetV2 model → class_suit
class_suit = models.mobilenet_v2(weights=None)
class_suit.classifier[1] = nn.Linear(class_suit.last_channel, 4)
class_suit = class_suit.to(device)
class_suit.load_state_dict(torch.load(SUIT_CLASSIFIER_PATH, map_location=device))
class_suit.eval()

# === SimpleOCRCNN model → class_cards
class SimpleOCRCNN(nn.Module):
    def __init__(self, num_classes=13):
        super(SimpleOCRCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 13)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

class_cards = SimpleOCRCNN(num_classes=13)
class_cards = class_cards.to(device)
class_cards.load_state_dict(torch.load(RANK_CLASSIFIER_PATH, map_location=device))
class_cards.eval()

# === Mappings
digit_class_map = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5',
                   6: '6', 7: '7', 8: '8', 9: '9', 10: '.'}

format_mapping = {0: "MTT", 1: "battle", 2: "battlef", 3: "spin"}

rank_mapping = {0: '2', 1: '3', 2: '4', 3: '5', 4: '6', 5: '7',
                6: '8', 7: '9', 8: '10', 9: 'J', 10: 'Q', 11: 'K', 12: 'A'}

suit_mapping = {0: 'spades', 1: 'hearts', 2: 'diamonds', 3: 'clubs'}

# === Transforms
format_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Main pipeline
def process_screen(image_cv):
    """
    Process poker table screenshot.

    Runs table format detection, YOLO object detection, and reconstructs full game state as JSON.

    Args:
        image_cv: OpenCV BGR image of the poker table screenshot.

    Returns:
        Dictionary with full parsed table state (format, players, hero cards, stacks, bets, ante, etc.).
    """

    image_bgr = image_cv.copy()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Determine format
    pil_img = Image.fromarray(image_rgb)
    img_t = format_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = class_format(img_t)
        _, preds = torch.max(outputs, 1)
        format_result = format_mapping[preds[0].item()]

    logging.info(f"Table format detected: {format_result}")

    # YOLO boxes predict
    results_boxes = yolo_boxes.predict(source=image_cv, imgsz=1600, conf=0.2, iou=0.5, device=device)[0]

    boxes = results_boxes.boxes.xyxy.cpu().numpy()
    classes = results_boxes.boxes.cls.cpu().numpy().astype(int)
    confidences = results_boxes.boxes.conf.cpu().numpy()

    # Debug print
    for box, cls, conf_score in zip(boxes, classes, confidences):
        logging.debug(f"YOLO class {cls} - conf={conf_score:.3f}")

    stacks_positions = []
    bets_candidates = []
    cards_found = []
    suits_found = []
    button_positions = []

    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)
        crop = image_bgr[y1:y2, x1:x2]
        center_x_px = (x1 + x2) / 2
        center_y_px = (y1 + y2) / 2

        if cls == 0:  # AllIn
            stacks_positions.append({
                "x": center_x_px,
                "y": center_y_px,
                "stack": 0.0
            })
            logging.debug(f"  ➡️ AllIn → Stack 0 at ({center_x_px:.1f}, {center_y_px:.1f})")
        elif cls == 1:  # Bets
            number = extract_number_from_crop(crop)
            bets_candidates.append({
                "x": center_x_px,
                "y": center_y_px,
                "value": number
            })
            logging.debug(f"  ➡️ Bet (candidate): {number} at ({center_x_px:.1f}, {center_y_px:.1f})")
        elif cls == 2:  # Button
            button_positions.append({"x": int(center_x_px), "y": int(center_y_px)})
            logging.debug(f"  🎯 Button found at: ({int(center_x_px)}, {int(center_y_px)})")
        elif cls == 3:  # Card Rank
            rank = classify_rank(crop, class_cards, device)
            cards_found.append({"x": center_x_px, "rank": rank})
            logging.debug(f"  🃏 Card Rank: {rank}")
        elif cls == 4:  # Stack
            number = extract_number_from_crop(crop)
            stacks_positions.append({
                "x": center_x_px,
                "y": center_y_px,
                "stack": number
            })
            logging.debug(f"  ➡️ Stack: {number} at ({center_x_px:.1f}, {center_y_px:.1f})")
        elif cls == 5:  # Suit
            suit = classify_suit(crop, class_suit, device)
            suits_found.append({"x": center_x_px, "suit": suit})
            logging.debug(f"  🃏 Card Suit: {suit}")

    # === Ante
    ante_value = None

    if format_result != "spin" and bets_candidates:
        target_x = image_bgr.shape[1] * 0.5
        target_y = image_bgr.shape[0] * 0.6

        def distance(cand):
            return ((cand["x"] - target_x) ** 2 + (cand["y"] - target_y) ** 2) ** 0.5

        bets_candidates_sorted = sorted(bets_candidates, key=distance)

        ante_value = bets_candidates_sorted[0]["value"]
        logging.debug(f"🟡 Ante selected: {ante_value}")

        # Remaining bets → assigned to player_bets
        bets_candidates = bets_candidates_sorted[1:]

    # === Button player idx
    def distance_xy(x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    button_player_idx = None
    if button_positions and stacks_positions:
        button_target_x = button_positions[0]["x"]
        button_target_y = button_positions[0]["y"]

        button_stack_idx = min(
            range(len(stacks_positions)),
            key=lambda i: distance_xy(
                stacks_positions[i]["x"], stacks_positions[i]["y"],
                button_target_x, button_target_y
            )
        )
        button_player_idx = button_stack_idx

        logging.debug(f"📍 Button coordinates: x={button_target_x:.1f}, y={button_target_y:.1f}")
    else:
        button_stack_idx = 0  # fallback

    # === Virtual "clockwise ray" for angle calculation
    def angle_from_center(x, y, center_x, center_y):
        return math.atan2(y - center_y, x - center_x)

    table_center_x = image_bgr.shape[1] * 0.5
    table_center_y = image_bgr.shape[0] * 0.5

    stack_angles = [angle_from_center(stack["x"], stack["y"], table_center_x, table_center_y)
                    for stack in stacks_positions]

    if not stack_angles or button_stack_idx >= len(stack_angles):
        raise ValueError("Invalid table: no button or stacks detected.")

    angle_Button = stack_angles[button_stack_idx]

    stack_angles_relative = []

    for idx, angle in enumerate(stack_angles):
        relative_angle = (angle - angle_Button + 2 * math.pi) % (2 * math.pi)
        stack_angles_relative.append({
            "idx": idx,
            "relative_angle": relative_angle
        })

    stack_angles_relative_sorted = sorted(stack_angles_relative, key=lambda x: x["relative_angle"])

    ordered_stack_indices = [x["idx"] for x in stack_angles_relative_sorted]

    shifted_ordered_stack_indices = ordered_stack_indices.copy()

    if len(ordered_stack_indices) == 2:
        shifted_ordered_stack_indices = ordered_stack_indices
    else:
        button_pos_in_current = ordered_stack_indices.index(button_stack_idx)
        target_pos = len(ordered_stack_indices) - 3
        shift = (button_pos_in_current - target_pos) % len(ordered_stack_indices)
        shifted_ordered_stack_indices = ordered_stack_indices[shift:] + ordered_stack_indices[:shift]

    # === Assign bets to players
    player_bets = {idx: 0.0 for idx in range(len(stacks_positions))}

    if bets_candidates:
        for bet in bets_candidates:
            nearest_stack_idx = min(range(len(stacks_positions)), key=lambda i: distance_xy(
                stacks_positions[i]["x"], stacks_positions[i]["y"],
                bet["x"], bet["y"]
            ))
            player_bets[nearest_stack_idx] += bet["value"]
            logging.debug(f"💰 Bet {bet['value']} → player_idx {nearest_stack_idx}")

    # === Hero cards → format like "6h", "6c"
    hero_cards_simple = []
    hero_present = False

    if len(cards_found) >= 2 and len(suits_found) >= 2:
        hero_present = True
        cards_sorted = sorted(cards_found, key=lambda x: x["x"])
        suits_sorted = sorted(suits_found, key=lambda x: x["x"])

        for i in range(2):
            rank = cards_sorted[i]["rank"]
            suit_letter = {
                "hearts": "h",
                "diamonds": "d",
                "clubs": "c",
                "spades": "s"
            }.get(suits_sorted[i]["suit"], "?")
            hero_cards_simple.append(f"{rank}{suit_letter}")

    # === Hero player idx → stack with maximum Y coordinate
    hero_player_idx = max(range(len(stacks_positions)), key=lambda i: stacks_positions[i]["y"])

    # === Determine player numbers
    button_player_num = None
    hero_player_num = None

    for player_num, stack_idx in enumerate(shifted_ordered_stack_indices, start=0):
        if stack_idx == button_player_idx:
            button_player_num = player_num
        if stack_idx == hero_player_idx:
            hero_player_num = player_num

    final_stacks = []

    for player_num, stack_idx in enumerate(shifted_ordered_stack_indices, start=0):
        stack_info = stacks_positions[stack_idx]
        final_stacks.append({
            "player": player_num,
            "stack": stack_info["stack"],
            "bet": player_bets[stack_idx]
        })

        final_json = {
        "format": format_result,
        "num_players": len(stacks_positions),
        "hero_present": hero_present,
        "button": button_player_num,
        "hero": hero_player_num,
        "hero_cards": hero_cards_simple,
        "ante": ante_value,
        "stacks": final_stacks
    }

    return final_json

