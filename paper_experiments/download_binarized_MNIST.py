import urllib.request
import os

urls = {
    "train": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/binarized_mnist_train.amat",
    "val": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/binarized_mnist_valid.amat",
    "test":  "https://storage.googleapis.com/tensorflow/tf-keras-datasets/binarized_mnist_test.amat",
}

if __name__ == "__main__":
    for split, url in urls.items():
        os.makedirs('./data/', exist_ok=True)
        fname = f"./data/binarized_mnist_{split}.amat"
        urllib.request.urlretrieve(url, fname)
        print(f"Downloaded {fname}")
