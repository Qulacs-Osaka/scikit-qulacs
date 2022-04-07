import pandas as pd
import pytest
from sklearn import datasets
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from skqulacs.circuit.pre_defined import create_qcl_ansatz
from skqulacs.qnn import QNNClassifier


@pytest.mark.parametrize(
    ("solver", "maxiter"), [("Adam_early_stopping", 777), ("BFGS", 8)]
)
def test_classify_iris(solver: str, maxiter: int):
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    x = df.loc[:, ["petal length (cm)", "petal width (cm)"]]
    x_train, x_test, y_train, y_test = train_test_split(
        x, iris.target, test_size=0.25, random_state=0
    )
    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()

    nqubit = 5
    c_depth = 3
    time_step = 0.5
    num_class = 3
    circuit = create_qcl_ansatz(nqubit, c_depth, time_step, 0)
    if solver == "Adam_early_stopping":
        qcl = QNNClassifier(circuit, num_class, "Adam", tol=0.01, n_iter_no_change=5)
    else:
        qcl = QNNClassifier(circuit, num_class, solver)

    qcl.fit(x_train, y_train, maxiter)
    y_pred = qcl.predict(x_test)
    assert f1_score(y_test, y_pred, average="weighted") > 0.92
