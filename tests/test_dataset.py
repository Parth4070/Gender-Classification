from src.dataset import create_datasets
from src.dataloader import create_dataloaders


DATA_DIR = "dataset"


train_dataset, validation_dataset, test_dataset = create_datasets(
    DATA_DIR
)


train_loader, validation_loader, test_loader = create_dataloaders(
    train_dataset,
    validation_dataset,
    test_dataset
)


print("Classes:", train_dataset.dataset.classes)

print("\nDataset sizes:")
print("Training:", len(train_dataset))
print("Validation:", len(validation_dataset))
print("Testing:", len(test_dataset))


images, labels = next(iter(train_loader))

print("\nOne batch:")
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)