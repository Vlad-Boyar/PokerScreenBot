
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from poker_pipeline.cards_preprocessing import preprocess_card_crop_opencv

# Define image transform for suit classifier
suit_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Mapping from class index to suit label
suit_mapping = {0: 'spades', 1: 'hearts', 2: 'diamonds', 3: 'clubs'}

# Mapping from class index to rank label
rank_mapping = {0: '2', 1: '3', 2: '4', 3: '5', 4: '6', 5: '7',
                6: '8', 7: '9', 8: '10', 9: 'J', 10: 'Q', 11: 'K', 12: 'A'}

# Classify card rank from image crop
def classify_rank(crop_img, class_cards, device):
    """
    Classify the rank of a card given its image crop.

    Args:
        crop_img: OpenCV image (BGR)
        class_cards: PyTorch model for rank classification
        device: PyTorch device (cpu or cuda)

    Returns:
        String with predicted rank label ('2', '3', ..., 'A')
    """
    crop_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    img_tensor = preprocess_card_crop_opencv(crop_gray)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = class_cards(img_tensor)
        _, predicted = torch.max(outputs, 1)
        cls_idx = predicted.item()
    return rank_mapping[cls_idx]

# Classify card suit from image crop
def classify_suit(crop_img, class_suit, device):
    """
    Classify the suit of a card given its image crop.

    Args:
        crop_img: OpenCV image (BGR)
        class_suit: PyTorch model for suit classification
        device: PyTorch device (cpu or cuda)

    Returns:
        String with predicted suit label ('spades', 'hearts', 'diamonds', 'clubs')
    """
    pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    img_tensor = suit_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = class_suit(img_tensor)
        _, predicted = torch.max(outputs, 1)
        cls_idx = predicted.item()
    return suit_mapping[cls_idx]
