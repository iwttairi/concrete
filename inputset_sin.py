# investigating the best inputset

from concrete import fhe
import numpy as np
import time

offset = 100


def sin(x):
    return (offset * np.sin(x / offset)).astype(np.int64)


compiler = fhe.Compiler(sin, {"x": "encrypted"})
# inputset = [-314, 314]
# inputset = [i for i in range(-314, 315)]
inputset = [i for i in np.linspace(-314, 314, 4, dtype=int)]
array = [i for i in range(-314, 315)]
print(f"Compiling...")
circuit = compiler.compile(inputset)

print(f"Generating keys...")
circuit.keygen()

answers = [(offset * np.sin(i / offset)).astype(np.int64) for i in array]

times = []
for i in range(10):
    examples = [j for j in range(-314, 315)]
    correct = 0
    error = []
    for j in range(len(examples)):
        encrypted_example = circuit.encrypt(examples[j])
        start_time = time.time()
        encrypted_result = circuit.run(encrypted_example)
        end_time = time.time()
        result = circuit.decrypt(encrypted_result)
        if answers[j] == result:
            correct += 1
        print(answers[j]/offset, result/offset)

        # excluding the first execution
        if not (i == 0 and examples[j] == -314):
            times.append(end_time-start_time)
    print(f'correct : {correct}')
    print(sum(times)/len(times), i)

print(sum(times)/len(times))

