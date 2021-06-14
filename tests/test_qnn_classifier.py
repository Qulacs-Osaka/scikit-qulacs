import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from skqulacs.classifier import QNNClassification


def test_classify_iris():
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    x = df.loc[:, ["petal width (cm)"]]
    x_train, x_test, y_train, y_test = train_test_split(
        x, iris.target, test_size=0.25, random_state=0
    )
    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()

    nqubit = 4  ## qubitの数。必要とする出力の次元数よりも多い必要がある
    c_depth = 3  ## circuitの深さ
    num_class = 3  ## 分類数（ここでは3つの品種に分類）
    qcl = QNNClassification(nqubit, c_depth, num_class)
    _, theta_opt = qcl.fit(x_train, y_train, maxiter=10)

    y_pred = qcl.predict(theta_opt, x_test)  # モデルのパラメータθも更新される
    assert f1_score(y_test, y_pred, average="weighted") > 0.95


if __name__ == "__main__":
    test_classify_iris()
