import numpy as np
import random
from numpy.random import RandomState
from skqulacs.qsvm import QSVR
from sklearn.metrics import mean_squared_error
from skqulacs.circuit import create_defqsv


def func_to_learn(x):
    return np.sin(x[0] * x[1] * 2)


def generate_noisy_sine(x_min: float, x_max: float, num_x: int):

    seed = 0
    random_state = RandomState(seed)

    x_train = []
    y_train = []
    for i in range(num_x):
        xa = x_min + (x_max - x_min) * random.random()
        xb = x_min + (x_max - x_min) * random.random()
        xc = 0
        xd = 0
        x_train.append([xa, xb, xc, xd])
        y_train.append(func_to_learn([xa, xb, xc, xd]))
        # 2要素だと量子的な複雑さが足りず、　精度が悪いため、ダミーの2bitを加えて4bitにしている。
    mag_noise = 0.05
    y_train += mag_noise * random_state.randn(num_x)
    return x_train, y_train


def test_noisy_sine():
    ########  パラメータ  #############

    x_min = -0.5
    x_max = 0.5
    num_x = 300
    x_train, y_train = generate_noisy_sine(x_min, x_max, num_x)

    n_qubit = 4
    circuit = create_defqsv(n_qubit, 4)
    qsvm = QSVR(circuit)
    qsvm.fit(x_train, y_train)
    loss = 0
    x_test, y_test = generate_noisy_sine(x_min, x_max, 100)
    y_pred = qsvm.predict(x_test)
    loss = mean_squared_error(y_pred, y_test)
    assert loss < 0.008


# 2要素のSVMを試してみる
# sin(x1*x2)をフィッティングさせる
def main():
    test_noisy_sine()


if __name__ == "__main__":
    main()
