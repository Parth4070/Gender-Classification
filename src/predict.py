from pathlib import Path

import torch
from PIL import Image

from .dataset import create_datasets
from .dataloader import create_dataloaders
from .model import create_resnet18
from .preprocessing import get_test_transforms


DATA_DIR = "dataset"
MODEL_PATH = Path("best_model/best_resnet18.pth")

CLASS_NAMES = [
    "male",
    "female"
]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")

    return torch.device("cpu")


def load_model():
    device = get_device()

    model = create_resnet18(
        num_classes=2,
        freeze_backbone=True
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model = model.to(device)

    model.eval()

    return model, device


def predict_image(image: Image.Image):
    model, device = load_model()

    transform = get_test_transforms()

    image = image.convert("RGB")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs,dim=1)

        confidence, predicted_class = torch.max(probabilities,dim=1)

    class_name = CLASS_NAMES[predicted_class.item()]

    confidence = confidence.item()

    return class_name, confidence