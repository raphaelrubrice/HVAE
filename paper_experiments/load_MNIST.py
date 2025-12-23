import tensorflow_datasets as tfds
import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import StratifiedKFold

# Dynamically binarize the images
def binarize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    # Dynamic binarization: sample from Bernoulli distribution
    image = tf.cast(tf.random.uniform(tf.shape(image)) < image, tf.float32)
    return image, label

def load_binarized_mnist_tensor(split="train", batch_size=128, shuffle=True):
    # Load all examples of the given split as a single batch
    ds = tfds.load('mnist', shuffle_files=True, split=split, batch_size=batch_size, as_supervised=True)
    # apply dynamic binarization as in Salakhutdinov & Murray, 2008
    ds = ds.map(binarize)
    ds_np = tfds.as_numpy(ds)
    # ds_np["image"]: shape (N, 28, 28), values {0,1}
    x = torch.from_numpy(np.concatenate([itm[0] for itm in ds_np])).float()
    y = torch.from_numpy(np.concatenate([itm[1] for itm in ds_np])).float()
    return x, y

def load_binarized_mnist_torch(split="train", batch_size=128, shuffle=True):
    x, y = load_binarized_mnist_tensor(split, batch_size, shuffle)

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle and (split == "train"))
    return loader

def get_loaders_MNIST():
    """
    Outputs the train, val and test data loaders for Binarized MNIST (2008)
    """
    train_loader = load_binarized_mnist_torch("train")
    test_loader  = load_binarized_mnist_torch("test", shuffle=False)
    return train_loader, test_loader

def make_cv_loaders_MNIST(cv=5, batch_size=128, **kwargs):
    """
    Loads the dynamically binarized MNIST trainset, applies
    a stratified K Fold CV split on it to form (train_loader, val_loader)
    pairs. Then loads the testset.
    Returns a list of (train_loader, val_loader) pairs for CV and the test loader
    """
    X, Y = load_binarized_mnist_tensor("train", batch_size=batch_size)

    skf = StratifiedKFold(n_splits=cv)
    cv_list = []
    for i, (train_index, val_index) in enumerate(skf.split(X, Y)):
        train_X, train_Y = X[train_index,:], Y[train_index]
        val_X, val_Y = X[val_index,:], Y[val_index]

        train_dataset = TensorDataset(train_X, train_Y) 
        train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            **kwargs)
        
        val_dataset = TensorDataset(val_X, val_Y) 
        val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            **kwargs)
        cv_list.append((train_loader, val_loader))

    test_loader  = load_binarized_mnist_torch("test", batch_size=batch_size, shuffle=False)
    return cv_list, test_loader

if __name__ == "__main__":
    print("Test loading MNIST")
    cv_list, test_loader = make_cv_loaders_MNIST()

    for train, val in cv_list:
        print(len(train), len(val))
    
    print(len(test_loader))