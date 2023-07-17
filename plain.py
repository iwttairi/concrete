# inference of plaintext dataset
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gzip
import os
import struct
import sys
import time

data_dir = "./data"

# MNISTデータセットの読み込みクラス
class MNISTDataset(Dataset):
    def __init__(self, data_dir, train=True):
        if train:
            images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
            labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        else:
            images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
            labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
        self.images, self.labels = self.load_data(images_path, labels_path)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    
    def load_data(self, images_path, labels_path):
        with gzip.open(labels_path, 'rb') as lbpath:
            _, n = struct.unpack('>II', lbpath.read(8))
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8)
        
        with gzip.open(images_path, 'rb') as imgpath:
            _, _, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), rows, cols)
        
        return (
            [torch.tensor(image, dtype=torch.float32, device="cuda:0") / 255.0 for image in images],
            [torch.tensor(label, dtype=torch.long, device="cuda:0") for label in labels]
        )


# loading test dataset
test_dataset = MNISTDataset(data_dir, train=False)
print(test_dataset[0])
print(type(test_dataset))
# making dataloader 
test_loader = DataLoader(test_dataset, shuffle=False)
# definition of the model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, device="cuda:0")
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size, device="cuda:0")

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

input_size = 28 * 28  # 入力の次元数（MNIST画像のサイズ）
hidden_size = 50  # 隠れ層のユニット数
output_size = 10  # 出力のクラス数（数字のクラス数）

# initialize the model
model = MLP(input_size, hidden_size, output_size)

# loading parameters
model.load_state_dict(torch.load('model_revised_199.pt' if len(sys.argv) < 2 else sys.argv[1]))

# evaluating the model
model.eval()
correct = 0
total = 0
execution_times = []
used_num = [0 for i in range(10)]
max_num = 10
i = 0
count = 0

# doing inference each label for 10 times
with torch.no_grad():
    for data, target in test_loader:
        if count == 10:
            break
        if i != 0:
            if used_num[target[0]] == max_num:
                i += 1
                continue
            else:
                used_num[target[0]] += 1
                if used_num[target[0]] == max_num:
                    count += 1
        start = time.time()
        output = model(data)
        end = time.time()
        if i != 0:
            execution_times.append(end - start)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            if predicted != target:
                print(output.data, target, i)
        i += 1

accuracy = correct / total
print(len(execution_times))
print(sum(execution_times) / len(execution_times))
print(f'Test Accuracy: {accuracy}')
