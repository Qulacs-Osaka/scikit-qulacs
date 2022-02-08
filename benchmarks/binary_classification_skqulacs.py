import pandas as pd
from sklearn import datasets
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from skqulacs.circuit.pre_defined import create_qcl_ansatz
from skqulacs.qnn import QNNClassifier


def binary_classification_skqulacs(n_qubit: int) -> float:
    """Solve a binary classification problem for the subset of iris dataset.

    Returns: F1 score for trained model.
    """
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    x = df.loc[:, ["petal length (cm)", "petal width (cm)"]]
    x = x.to_numpy()
    # Now we are going to solve binary classification, so exclude the third class.
    index = iris.target != 2
    x = x[index]
    y = iris.target[index]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=0
    )

    circuit_depth = 3
    time_step = 0.5
    num_class = 2
    circuit = create_qcl_ansatz(n_qubit, circuit_depth, time_step, 0)
    qcl = QNNClassifier(circuit, num_class, "Adam")
    qcl.fit(x_train, y_train, 7)
    y_pred = qcl.predict(x_test)
    return f1_score(y_test, y_pred)
