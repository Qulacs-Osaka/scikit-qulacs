import numpy as np
import matplotlib.pyplot as plt
from qnn import QNNRegressor


def main():
    ########  パラメータ  #############
    nqubit = 3  ## qubitの数
    c_depth = 4  ## circuitの深さ
    time_step = 0.77  ## ランダムハミルトニアンによる時間発展の経過時間

    ## [x_min, x_max]のうち, ランダムにnum_x_train個の点をとって教師データとする.
    x_min = -1.0
    x_max = 1.0
    num_x_train = 50

    ## 学習したい1変数関数
    func_to_learn = lambda x: np.sin(x * np.pi)

    ## 乱数のシード
    random_seed = 0
    ## 乱数発生器の初期化
    np.random.seed(random_seed)
    x_train = x_min + (x_max - x_min) * np.random.rand(num_x_train)
    y_train = func_to_learn(x_train)

    # 現実のデータを用いる場合を想定し、きれいなsin関数にノイズを付加
    mag_noise = 0.05
    y_train = y_train + mag_noise * np.random.randn(num_x_train)

    qnn = QNNRegressor(nqubit, c_depth, time_step)
    res, theta_opt = qnn.fit(x_train, y_train)
    print(res)

    xlist = np.arange(x_min, x_max, 0.02)
    # モデルの予測値
    y_pred = np.array([qnn.predict(x) for x in xlist])

    plt.figure(figsize=(10, 6))
    plt.plot(x_train, y_train, "o", label="Teacher")
    plt.plot(xlist, y_pred, label="Final Model Prediction")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
