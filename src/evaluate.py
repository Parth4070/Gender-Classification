import torch
from pathlib import Path
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix)

from .dataset import create_datasets
from .dataloader import create_dataloaders
from .model import create_resnet18


DATA_DIR = "dataset"
MODEL_PATH = Path("best_model/best_resnet18.pth")

BATCH_SIZE = 16


def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")

    return torch.device("cpu")


def evaluate_model(model,test_loader,device):

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

    return all_labels, all_predictions


def main():

    device = get_device()

    print(f"Using device: {device}")

    _, _, test_dataset = create_datasets(
        DATA_DIR
    )

    _, _, test_loader = create_dataloaders(
        None,
        None,
        test_dataset,
        batch_size=BATCH_SIZE
    )

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

    labels, predictions = evaluate_model(model,test_loader,device)

    accuracy = accuracy_score(labels,predictions)

    print("\nTest Accuracy:")
    print(f"{accuracy:.4f}")

    print("\nClassification Report:")

    print(classification_report(labels,predictions,target_names=["men", "women"],digits=4))

    print("Confusion Matrix:")

    print(confusion_matrix(labels,predictions))


if __name__ == "__main__":
    main()