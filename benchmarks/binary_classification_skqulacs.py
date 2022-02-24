import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from qulacs.gate import Pauli
from numpy.random import default_rng, Generator

from skqulacs.qnn import QNNClassifier
from skqulacs.circuit import LearningCircuit


def get_angles(x: np.ndarray) -> np.ndarray:
    beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
    beta1 = 2 * np.arcsin(np.sqrt(x[3] ** 2) / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
    beta2 = 2 * np.arcsin(
        np.sqrt(x[2] ** 2 + x[3] ** 2)
        / np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2)
    )

    return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])


def preprocess_input(x: np.ndarray) -> np.ndarray:
    padding = 0.3 * np.ones((len(x), 1))
    X_pad = np.c_[np.c_[x, padding], np.zeros((len(x), 1))]

    # normalize each input
    normalization = np.sqrt(np.sum(X_pad ** 2, -1))
    X_norm = (X_pad.T / normalization).T

    # angles for state preparation are new features
    return np.array([get_angles(x) for x in X_norm])


def create_circuit(depth: int) -> LearningCircuit:
    """
    Create a circuit which is equivalent to the one introduced in PennyLane's tutorial:
    https://pennylane.ai/qml/demos/tutorial_variational_classifier.html
    """

    def prepare_state(circuit: LearningCircuit) -> None:
        circuit.add_input_RY_gate(0)

        circuit.add_CNOT_gate(0, 1)
        circuit.add_input_RY_gate(1)
        circuit.add_CNOT_gate(0, 1)
        circuit.add_input_RY_gate(1)

        circuit.add_gate(Pauli([0], [1]))  # Pauli X on 0th qubit
        circuit.add_CNOT_gate(0, 1)
        circuit.add_input_RY_gate(1)
        circuit.add_CNOT_gate(0, 1)
        circuit.add_input_RY_gate(1)
        circuit.add_gate(Pauli([0], [1]))

    def add_layer(circuit: LearningCircuit, rng: Generator) -> None:
        circuit.add_parametric_RZ_gate(0, rng.random())
        circuit.add_parametric_RX_gate(0, rng.random())
        circuit.add_parametric_RZ_gate(0, rng.random())

        circuit.add_parametric_RZ_gate(1, rng.random())
        circuit.add_parametric_RX_gate(1, rng.random())
        circuit.add_parametric_RZ_gate(1, rng.random())

        circuit.add_CNOT_gate(0, 1)

    n_qubit = 2
    circuit = LearningCircuit(n_qubit)

    prepare_state(circuit)

    rng = default_rng(0)
    for _ in range(depth):
        add_layer(circuit, rng)

    return circuit

def binary_classification_skqulacs(_n_qubit: int) -> float:
    """Solve a binary classification problem for the subset of iris dataset.

    Returns: F1 score for trained model.
    """
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    x = df.loc[:, ["petal length (cm)", "petal width (cm)"]]
    x = x.to_numpy()
    # Now we are going to solve binary classification, so exclude the third class.
    index = iris.target != 2
    x = preprocess_input(x[index])
    y = iris.target[index]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=0
    )

    num_class = 2
    circuit = create_circuit(6)
    qcl = QNNClassifier(circuit, num_class, "Nelder-Mead", do_x_scale=False)
    qcl.fit(x_train, y_train, 1000)
    y_pred = qcl.predict(x_test)
    return f1_score(y_test, y_pred)


if __name__ == "__main__":
    score = binary_classification_skqulacs(2)
    print(score)
