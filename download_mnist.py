# downloading the mnist dataset

import os
import urllib.request
import gzip
import numpy as np

def download_mnist(save_dir):
    # URL of mnist
    base_url = "http://yann.lecun.com/exdb/mnist/"
    file_names = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]

    os.makedirs(save_dir, exist_ok=True)

    # downloading files
    for file_name in file_names:
        url = base_url + file_name
        save_path = os.path.join(save_dir, file_name)
        urllib.request.urlretrieve(url, save_path)

    print("MNIST dataset downloaded!")

def load_mnist(data_dir, train=True):
    if train:
        images_path = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
        labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    else:
        images_path = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
        labels_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")

    with gzip.open(images_path, "rb") as imgpath, gzip.open(labels_path, "rb") as lbpath:
        images = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(-1, 28, 28)
        labels = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    return images, labels

# downloading and loading mnist dataset
data_dir = "./data"
download_mnist(data_dir)
train_images, train_labels = load_mnist(data_dir, train=True)
