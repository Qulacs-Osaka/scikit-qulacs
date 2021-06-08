import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets

from classifier import QNNClassification

def test_classify_iris():
    iris = datasets.load_iris()

    # 扱いやすいよう、pandasのDataFrame形式に変換
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['target_names'] = iris.target_names[iris.target]

    ## 教師データ作成
    # ここではpetal length, petal widthの2種類のデータを用いる。より高次元への拡張は容易である。
    x_train = df.loc[:,['petal length (cm)', 'petal width (cm)']].to_numpy() # shape:(150, 2)
    y_train = np.eye(3)[iris.target] # one-hot 表現 shape:(150, 3)

    nqubit = 4 ## qubitの数。必要とする出力の次元数よりも多い必要がある
    c_depth = 3 ## circuitの深さ
    num_class = 3 ## 分類数（ここでは3つの品種に分類）
    qcl = QNNClassification(nqubit, c_depth, num_class)
    loss, theta_opt = qcl.fit(x_train, y_train, maxiter=10)
    h = .05  # step size in the mesh
    x_min, x_max = x_train[:, 0].min() - .5, x_train[:, 0].max() + .5
    y_min, y_max = x_train[:, 1].min() - .5, x_train[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    qcl._set_input_state(np.c_[xx.ravel(), yy.ravel()])
    Z = qcl.predict(theta_opt) # モデルのパラメータθも更新される
    Z = np.argmax(Z, axis=1)

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 5))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.scatter(x_train[:, 0], x_train[:, 1], c=iris.target, edgecolors='k', cmap=plt.cm.Paired)

    # label
    plt.title('optimized')
    plt.xlabel('petal length (cm)')
    plt.ylabel('petal width (cm)')
    plt.show()

if __name__ == "__main__":
    test_classify_iris()