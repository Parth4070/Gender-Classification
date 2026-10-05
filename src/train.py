import torch
import torch.nn as nn
from pathlib import Path

from .dataset import create_datasets
from .dataloader import create_dataloaders
from .model import create_resnet18

DATA_DIR = "dataset"
MODEL_DIR = Path("best_model")

BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")

    return torch.device("cpu")

device = get_device()

print(f"Using device: {device}")

train_dataset, validation_dataset, test_dataset = create_datasets(DATA_DIR)

train_loader, validation_loader, test_loader = create_dataloaders(
    train_dataset,
    validation_dataset,
    test_dataset,
    batch_size=BATCH_SIZE
)

model = create_resnet18()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    filter(
        lambda parameter:parameter.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

def train_one_epoch(model,loader,criterion,optimizer,device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy

def validate(model,loader,criterion,device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def main():
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    device = get_device()
    train_dataset, validation_dataset, test_dataset = create_datasets(
        DATA_DIR
    )

    train_loader, validation_loader, test_loader = create_dataloaders(
        train_dataset,
        validation_dataset,
        test_dataset,
        batch_size=BATCH_SIZE
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")

    model = create_resnet18(num_classes=2,freeze_backbone=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters()
        ),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    best_validation_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        validation_loss, validation_accuracy = validate(
            model,
            validation_loader,
            criterion,
            device
        )

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy:.4f} "
            f"Val Loss: {validation_loss:.4f} "
            f"Val Acc: {validation_accuracy:.4f}"
        )

        if validation_accuracy > best_validation_accuracy:

            best_validation_accuracy = validation_accuracy

            torch.save(
                model.state_dict(),
                MODEL_DIR / "best_resnet18.pth"
            )

            print(
                f"Saved best model "
                f"(Val Acc: {validation_accuracy:.4f})"
            )


if __name__ == "__main__":
    main()