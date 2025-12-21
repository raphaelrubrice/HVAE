import tensorflow_datasets as tfds
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def load_binarized_mnist_torch(split="train", batch_size=128, shuffle=True):
    # Load all examples of the given split as a single batch
    ds = tfds.load("binarized_mnist",
                   split=split,
                   batch_size=-1,  # everything at once
                   as_supervised=False)

    ds_np = tfds.as_numpy(ds)
    # ds_np["image"]: shape (N, 28, 28), values {0,1}
    x = torch.from_numpy(ds_np["image"]).float()
    dataset = TensorDataset(x)  # no labels in this benchmark
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle and (split == "train"))
    return loader

def get_loaders_MNIST():
    """
    Outputs the train, val and test data loaders for Binarized MNIST (2008)
    """
    train_loader = load_binarized_mnist_torch("train")
    val_loader = load_binarized_mnist_torch("validation", shuffle=False)
    test_loader  = load_binarized_mnist_torch("test", shuffle=False)
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    print("Test loading MNIST")
    train, val, test = get_loaders_MNIST()

    print(f"Train size: {len(train)}")
    print(len(next(iter(train))), next(iter(train))[0].size())
    print(f"Val size: {len(val)}")
    print(f"Test size: {len(test)}")