
import numpy as np
from concrete import fhe
import torch
import time

data_dir = "./data"

# loading data and model
test_images = torch.load('./data/test_images.pt')
test_labels = torch.load('./data/test_labels.pt')
model = torch.load("model_revised_199.pt")
offset = 2

# loading parameters
fc1_weight = model['fc1.weight'].detach().cpu().numpy()
fc1_bias = model['fc1.bias'].detach().cpu().numpy()
fc2_weight = model['fc2.weight'].detach().cpu().numpy()
fc2_bias = model['fc2.bias'].detach().cpu().numpy()

# changing parameters into integer
fc1_weight = (fc1_weight * offset).astype(np.int64)
fc1_bias = (fc1_bias * offset * offset).astype(np.int64)
fc2_weight = (fc2_weight * offset).astype(np.int64)
fc2_bias = (fc2_bias * offset * offset).astype(np.int64)


images = [tensor.detach().cpu().numpy() for tensor in test_images]
labels = [tensor.detach().cpu().numpy() for tensor in test_labels]

def linear1(x):
    return x @ fc1_weight.T + fc1_bias

def linear2(x):
    x = x//offset
    return x @ fc2_weight.T + fc2_bias

def relu(x):
    return np.maximum(x, 0)

# def mlp(x):
#     x = linear1(x)
#     x = (x//offset)
#     x = relu(x)
#     x = linear2(x)
#     return x

# compiling each function
compiler1 = fhe.Compiler(linear1, {"x": "encrypted"})
compiler2 = fhe.Compiler(relu, {"x": "encrypted"})
compiler3 = fhe.Compiler(linear2, {"x": "encrypted"})

# inputsets
inputset1 = [(np.array([0 for i in range(784)])), (np.array([offset for i in range(784)]))]
inputset2 = [(np.array([-606 for i in range(50)])), (np.array([489 for i in range(50)]))]
inputset3 = [(np.array([0 for i in range(50)])), (np.array([489 for i in range(50)]))]

print(f"Compiling...")
circuit1 = compiler1.compile(inputset1, verbose=True, show_mlir=True)
circuit2 = compiler2.compile(inputset2, verbose=True, show_mlir=True)
circuit3 = compiler3.compile(inputset3, verbose=True, show_mlir=True)

print(f"Generating keys...")
circuit1.keygen()
circuit2.keygen()
circuit3.keygen()

input_size = 28 * 28  # 入力の次元数（MNIST画像のサイズ）
hidden_size = 50  # 隠れ層のユニット数
output_size = 10  # 出力のクラス数（数字のクラス数）

correct = 0
total = 0

executiontime1 = []
executiontime2 = []
executiontime3 = []

for i in range(101): 
    data = images[i]
    target = labels[i]
    data = data.reshape(1, input_size)
    data *= offset
    data = data[0]
    data = data.astype(np.int64)
# excuting linear1
    encrypted_data = circuit1.encrypt(data)
    start = time.time()
    encrypted_output1 = circuit1.run(encrypted_data)
    end = time.time()
    if i != 0:
        executiontime1.append(end-start)
    decrypted_data1 = circuit1.decrypt(encrypted_output1)
# excuting relu
    encrypted_data2 = circuit2.encrypt(decrypted_data1)
    start = time.time()
    encrypted_output2 = circuit2.run(encrypted_data2)
    end = time.time()
    if i != 0:
        executiontime2.append(end-start)
    decrypted_data2 = circuit2.decrypt(encrypted_output2)
# excuting linear2
    encrypted_data3 = circuit3.encrypt(decrypted_data2)
    start = time.time()
    encrypted_output3 = circuit3.run(encrypted_data3)
    end = time.time()
    if i != 0:
        executiontime3.append(end-start)
    output = circuit3.decrypt(encrypted_output3)

print(f'Linear1 execution time: {sum(executiontime1)/len(executiontime1)}')
print(f'Relu execution time: {sum(executiontime2)/len(executiontime2)}')
print(f'Linear2 execution time: {sum(executiontime3)/len(executiontime3)}')