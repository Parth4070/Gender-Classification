from torch.utils.data import DataLoader


def create_dataloaders(train_dataset=None, validation_dataset=None, test_dataset=None, batch_size=16, num_workers=0):
    train_loader = None
    validation_loader = None
    test_loader = None

    if train_dataset is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
    )

    if validation_dataset is not None:
        validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    if test_dataset is not None:
        test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader,validation_loader,test_loader