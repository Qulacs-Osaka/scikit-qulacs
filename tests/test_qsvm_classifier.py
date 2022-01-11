import pandas as pd
from sklearn import datasets
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from skqulacs.circuit import create_ibm_embedding_circuit
from skqulacs.qsvm import QSVC


def test_classify_iris():
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    x = df.loc[:, ["petal length (cm)", "petal width (cm)"]]
    x_train, x_test, y_train, y_test = train_test_split(
        x, iris.target, test_size=0.25, random_state=1
    )

    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()
    n_qubit = 4  # qubitの数 2だと少なすぎて複雑さがでない
    circuit = create_ibm_embedding_circuit(n_qubit)
    qsvm = QSVC(circuit)
    qsvm.fit(x_train, y_train)
    y_pred = qsvm.predict(x_test)
    assert f1_score(y_test, y_pred, average="weighted") > 0.92


if __name__ == "__main__":
    test_classify_iris()
