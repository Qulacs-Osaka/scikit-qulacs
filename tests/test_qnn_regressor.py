import numpy as np
import random
from numpy.random import RandomState
from skqulacs.qnn import QNNRegressor
from sklearn.metrics import mean_squared_error


def func_to_learn(x):
    return np.sin(x[0] * x[1] * np.pi)


def generate_noisy_sine(x_min: float, x_max: float, num_x: int):

    seed = 0
    random_state = RandomState(seed)

    x_train = []
    y_train = []
    for i in range(num_x):
        xa = x_min + (x_max - x_min) * random.random()
        xb = x_min + (x_max - x_min) * random.random()
        x_train.append([xa, xb])
        y_train.append(func_to_learn([xa, xb]))
        # x_train.append([xa])
        # y_train.append(func_to_learn([xa, 1]))
        # 2要素だと量子的な複雑さが足りず、　精度が悪いため、ダミーの2bitを加えて4bitにしている。
    # print(x_train)
    # print(y_train)
    mag_noise = 0.0005
    y_train += mag_noise * random_state.randn(num_x)
    return x_train, y_train


def test_noisy_sine():
    ########  パラメータ  #############

    x_min = -0.5
    x_max = 0.5
    num_x = 100
    x_train, y_train = generate_noisy_sine(x_min, x_max, num_x)

    n_qubit = 4
    depth = 3
    qnn = QNNRegressor(n_qubit, depth)
    qnn.fit(x_train, y_train, maxiter=1000)
    loss = 0
    x_test, y_test = generate_noisy_sine(x_min, x_max, 100)
    y_pred = qnn.predict(x_test)
    loss = mean_squared_error(y_pred, y_test)
    print(loss)
    for i in range(len(x_test)):
        print([x_test[i][0], x_test[i][1], y_test[i], y_pred[i]])
    assert loss < 0.05


# 2要素のQNNを試してみる
# sin(x1*x2)をフィッティングさせる
def main():
    test_noisy_sine()


if __name__ == "__main__":
    main()
