import numpy as np
import pytest
from numpy.random import default_rng
from sklearn.metrics import f1_score

from skqulacs.circuit.pre_defined import create_qcnn_ansatz
from skqulacs.qnn import QNNClassifier
from skqulacs.qnn.solver import Adam, Solver


def generate_data(bits: int, random_seed: int = 0):
    """Generate training and testing data."""
    rng = default_rng(random_seed)
    n_rounds = 20  # Produces n_rounds * bits datapoints.
    excitations = []
    labels = []
    for _ in range(n_rounds):
        for _ in range(bits):
            r = rng.uniform(-np.pi, np.pi)
            excitations.append(r)
            labels.append(1 if (-np.pi / 2) <= r <= (np.pi / 2) else 0)

    train_ratio = 0.7
    split_ind = int(len(excitations) * train_ratio)
    train_excitations = excitations[:split_ind]
    test_excitations = excitations[split_ind:]

    train_labels = labels[:split_ind]
    test_labels = labels[split_ind:]

    return (
        np.array(train_excitations),
        np.array(train_labels),
        np.array(test_excitations),
        np.array(test_labels),
    )


@pytest.mark.parametrize(("solver", "maxiter"), [(Adam(), 20)])
def test_qcnn(solver: Solver, maxiter: int):
    n_qubit = 8
    random_seed = 0
    circuit = create_qcnn_ansatz(n_qubit, random_seed)

    num_class = 2
    qcl = QNNClassifier(circuit, num_class, solver)

    x_train, y_train, x_test, y_test = generate_data(n_qubit)
    qcl.fit(x_train, y_train, maxiter)
    y_pred = qcl.predict(x_test)
    score = f1_score(y_test, y_pred, average="weighted")
    assert score > 0.9
