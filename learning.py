import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import gzip
import os
import struct
import numpy as np
import torch.nn.functional as F

data_dir = './data'

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

    
# loading the train dataset
train_dataset = MNISTDataset(data_dir, train=True)

batch_size = 100
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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
        x = F.softmax(x, dim=1)
        return x

input_size = 28 * 28  # 入力の次元数（MNIST画像のサイズ）
hidden_size = 50  # 隠れ層のユニット数
output_size = 10  # 出力のクラス数（数字のクラス数）
learning_rate = 0.01
num_epochs = 200


model = MLP(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# training
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx+1}/{len(train_loader)}, Loss: {loss.item()}')

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), './model_revised_{}.pt'.format(epoch))
