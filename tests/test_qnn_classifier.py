import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from skqulacs.classifier import QNNClassification


def test_classify_iris():
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    x = df.loc[:, ["petal length (cm)", "petal width (cm)"]]
    x_train, x_test, y_train, y_test = train_test_split(
        x, iris.target, test_size=0.25, random_state=0
    )
    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()
    y_train = np.eye(3)[y_train]  # one-hot 表現 shape:(150, 3)

    nqubit = 4  ## qubitの数。必要とする出力の次元数よりも多い必要がある
    c_depth = 3  ## circuitの深さ
    num_class = 3  ## 分類数（ここでは3つの品種に分類）
    qcl = QNNClassification(nqubit, c_depth, num_class)
    loss, theta_opt = qcl.fit(x_train, y_train, maxiter=10)

    # h = .05  # step size in the mesh
    # x_min, x_max = x_test[:, 0].min() - .5, x_test[:, 0].max() + .5
    # y_min, y_max = x_test[:, 1].min() - .5, x_test[:, 1].max() + .5
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    # qcl._set_input_state(np.c_[xx.ravel(), yy.ravel()])
    y_pred = qcl.predict(theta_opt, x_test)  # モデルのパラメータθも更新される
    y_pred = list(map(np.argmax, y_pred))
    print(classification_report(y_test, y_pred))
    assert False
    # print(list(zip(Z, y_test)))
    # Z = np.argmax(Z, axis=1)

    # # Put the result into a color plot
    # Z = Z.reshape(xx.shape)
    # plt.figure(figsize=(8, 5))
    # plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # # Plot also the training points
    # plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, edgecolors='k', cmap=plt.cm.Paired)

    # # label
    # plt.title('optimized')
    # plt.xlabel('petal length (cm)')
    # plt.ylabel('petal width (cm)')
    # plt.savefig("classification.png")


if __name__ == "__main__":
    test_classify_iris()
