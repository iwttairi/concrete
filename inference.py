# inference of encrypted data
import numpy as np
from concrete import fhe
import torch
import time

data_dir = "./data"

label = {}

# loading parameters and data
test_images = torch.load('./data/test_images.pt')
test_labels = torch.load('./data/test_labels.pt')
model = torch.load("model_revised_199.pt")
offset = 2
fc1_weight = model['fc1.weight'].detach().cpu().numpy()
fc1_bias = model['fc1.bias'].detach().cpu().numpy()
fc2_weight = model['fc2.weight'].detach().cpu().numpy()
fc2_bias = model['fc2.bias'].detach().cpu().numpy()
fc1_weight = (fc1_weight * offset).astype(np.int64)
fc1_bias = (fc1_bias * offset * offset).astype(np.int64)
fc2_weight = (fc2_weight * offset).astype(np.int64)
fc2_bias = (fc2_bias * offset * offset).astype(np.int64)


images = [tensor.detach().cpu().numpy() for tensor in test_images]
labels = [tensor.detach().cpu().numpy() for tensor in test_labels]

def linear1(x):
    return x @ fc1_weight.T + fc1_bias

def linear2(x):
    return x @ fc2_weight.T + fc2_bias

def relu(x):
    return np.maximum(x, 0)

def mlp(x):
    x = linear1(x)
    x = (x//offset)
    x = relu(x)
    x = linear2(x)
    return x

# compiling mlp
compiler = fhe.Compiler(mlp, {"x": "encrypted"})

inputset = [(np.array([0 for i in range(784)])), (np.array([offset for i in range(784)]))]

print(f"Compiling...")
circuit = compiler.compile(inputset, verbose=True, show_mlir=True)

print(f"Generating keys...")
circuit.keygen()

input_size = 28 * 28  # 入力の次元数（MNIST画像のサイズ）
hidden_size = 50  # 隠れ層のユニット数
output_size = 10  # 出力のクラス数（数字のクラス数）

correct = 0
total = 0

execution_times = []
max_num = 10
count = 0
used_num = [0 for i in range(10)]

# doing inference each label for 10 times
for i in range(10000): 
    data = images[i]
    target = labels[i]
    if used_num[target] == max_num:
        count += 1
        continue
    if sum(used_num) == max_num * 10:
        break
    elif i != 0:
        used_num[target] += 1
    data = data.reshape(1, input_size)
    data *= offset
    data = data[0]
    data = data.astype(np.int64)
    encrypted_data = circuit.encrypt(data)
    start = time.time()
    encrypted_output = circuit.run(encrypted_data)
    end = time.time()
    output = circuit.decrypt(encrypted_output)
    predicted = np.argmax(output)
    # excluding the first execution
    if i != 0:
        total += 1
        if predicted == target:
            correct += 1
        else:
            print(predicted, target, output, i)
        execution_times.append(end-start)


accuracy = correct / total
print(f'Test Accuracy: {accuracy}')
print(len(execution_times), total, correct)
print(f'Average Execution Time: {sum(execution_times)/len(execution_times)}')