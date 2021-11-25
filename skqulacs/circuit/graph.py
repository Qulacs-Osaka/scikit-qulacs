# 回路のグラフ化をできます。量子状態の棒グラフ、縮約した後の玉表示
import matplotlib.pyplot as plt
import numpy as np
from qulacs_osaka import Observable


def show_blochsphere(state, bit):
    n_qubit = state.get_qubit_count()
    observableX = Observable(n_qubit)
    observableX.add_operator(1.0, f"X {bit}")  # オブザーバブルを設定
    observableY = Observable(n_qubit)
    observableY.add_operator(1.0, f"Y {bit}")  # オブザーバブルを設定
    observableZ = Observable(n_qubit)
    observableZ.add_operator(1.0, f"Z {bit}")  # オブザーバブルを設定

    X = observableX.get_expectation_value(state)
    Y = observableY.get_expectation_value(state)
    Z = observableZ.get_expectation_value(state)
    print(X, Y, Z)
    fig = plt.figure()

    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1, 1, 1))
    # sphere
    u, v = np.mgrid[0 : (2 * np.pi) : 8j, 0 : np.pi : 8j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="lightskyblue", linewidth=0.5)
    ax.quiver(0, 0, 0, X, Y, Z, color="red")
    ax.scatter(X, Y, Z, color="red")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    plt.show()
