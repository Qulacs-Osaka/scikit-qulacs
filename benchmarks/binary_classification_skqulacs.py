from typing import Tuple

import numpy as np
import pandas as pd
from binary_classification_common import preprocess_input
from numpy.random import Generator, default_rng
from qulacs.gate import Pauli
from sklearn import datasets
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from skqulacs.circuit import LearningCircuit
from skqulacs.qnn import QNNClassifier


def load_iris_skqulacs() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    # Discard the half of the test data because in PennyLane implementation, 25% of data
    # are used for validation, which is not needed for skqulacs's implementation.
    _, x_test, _, y_test = train_test_split(
        x_test, y_test, test_size=0.5, random_state=0
    )

    return x_train, x_test, y_train, y_test


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


def binary_classification_skqulacs(
    circuit: LearningCircuit,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Solve a binary classification problem for the subset of iris dataset.

    Returns: F1 score for trained model.
    """

    num_class = 2
    circuit = create_circuit(6)
    qcl = QNNClassifier(circuit, num_class, "Adam", do_x_scale=False)
    qcl.fit(x_train, y_train, 50)
    y_pred = qcl.predict(x_test)
    f1_score(y_test, y_pred) > 0.95
