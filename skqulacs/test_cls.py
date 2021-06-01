import numpy as np
from qulacs import QuantumState, Observable, QuantumCircuit, ParametricQuantumCircuit
from sklearn.metrics import log_loss
from scipy.optimize import minimize

import matplotlib.pyplot as plt

from functools import reduce
from qulacs.gate import X, Z, DenseMatrix
import pandas as pd
from sklearn import datasets
import time

import qnn
from qnn import QNNClassification

def main():



    iris = datasets.load_iris()

    # 扱いやすいよう、pandasのDataFrame形式に変換
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['target_names'] = iris.target_names[iris.target]
    df.head()
    # サンプル総数
    print(f"# of records: {len(df)}\n")

    # 各品種のサンプル数
    print("value_counts:")
    print(df.target_names.value_counts())
    ## 教師データ作成
    # ここではpetal length, petal widthの2種類のデータを用いる。より高次元への拡張は容易である。
    x_train = df.loc[:,['petal length (cm)', 'petal width (cm)']].to_numpy() # shape:(150, 2)
    y_train = np.eye(3)[iris.target] # one-hot 表現 shape:(150, 3)
    plt.figure(figsize=(8, 5))

    for t in range(3):
        x = x_train[iris.target==t][:,0]
        y = x_train[iris.target==t][:,1]
        cm = [plt.cm.Paired([c]) for c in [0,6,11]]
        plt.scatter(x, y, c=cm[t], edgecolors='k', label=iris.target_names[t])

    # label
    plt.title('Iris dataset')
    plt.xlabel('petal length (cm)')
    plt.ylabel('petal width (cm)') 
    plt.legend()
    plt.show()

    random_seed = 0
    np.random.seed(random_seed)
    nqubit = 4 ## qubitの数。必要とする出力の次元数よりも多い必要がある
    c_depth = 4 ## circuitの深さ
    num_class = 3 ## 分類数（ここでは3つの品種に分類）
    qcl = QNNClassification(nqubit, c_depth, num_class)
    start = time.time()
    res, theta_init, theta_opt = qcl.fit(x_train, y_train, maxiter=10)
    print(f'elapsed time: {time.time() - start:.1f}s')
    h = .05  # step size in the mesh
    X = x_train
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    plt.figure(figsize=(8, 5))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    qcl.set_input_state(np.c_[xx.ravel(), yy.ravel()])
    Z = qcl.pred(qcl.theta) # モデルのパラメータθも更新される
    Z = np.argmax(Z, axis=1)
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=iris.target, edgecolors='k', cmap=plt.cm.Paired)

    # label
    plt.title('optimized')
    plt.xlabel('petal length (cm)')
    plt.ylabel('petal width (cm)')
    plt.show()

if __name__ == "__main__":
    main()