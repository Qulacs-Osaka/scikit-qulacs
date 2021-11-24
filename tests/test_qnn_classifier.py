import pytest
from skqulacs.circuit.pre_defined import create_qcl_ansatz
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from skqulacs.qnn import QNNClassifier

# ("Nelder-Mead", 500)



@pytest.mark.parametrize(("solver", "maxiter"), [("Adam", 7), ("BFGS", 8)])
def test_classify_iris(solver: str, maxiter: int):
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    x = df.loc[:, ["petal length (cm)", "petal width (cm)"]]
    x_train, x_test, y_train, y_test = train_test_split(
        x, iris.target, test_size=0.25, random_state=0
    )
    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()

    nqubit = 5  # qubitの数。必要とする出力の次元数以上の必要がある
    c_depth = 3  # circuitの深さ
    time_step = 0.5
    num_class = 3  # 分類数（ここでは3つの品種に分類）
    circuit = create_qcl_ansatz(nqubit, c_depth, time_step, 0)
    qcl = QNNClassifier(nqubit, circuit, num_class, solver)

    for kai in range(maxiter):
        qcl.fit(x_train, y_train, 1)
        y_pred = qcl.predict(x_test)
        print(f1_score(y_test, y_pred, average="weighted"))
    assert f1_score(y_test, y_pred, average="weighted") > 0.92
