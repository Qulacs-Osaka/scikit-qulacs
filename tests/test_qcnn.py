import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from sklearn.metrics import f1_score
from skqulacs.qnn import QNNClassifier
from skqulacs.circuit.pre_defined import create_qcnn_ansatz
import pytest


def generate_data(bits: int, random_seed: int = 0):
    """Generate training and testing data."""
    rng = default_rng(random_seed)
    n_rounds = 20  # Produces n_rounds * bits datapoints.
    excitations = []
    labels = []
    for n in range(n_rounds):
        for bit in range(bits):
            r = rng.uniform(-np.pi, np.pi)
            excitations.append(r)
            labels.append(1 if (-np.pi / 2) <= r <= (np.pi / 2) else 0)

    split_ind = int(len(excitations) * 0.7)
    train_excitations = excitations[:split_ind]
    test_excitations = excitations[split_ind:]

    train_labels = labels[:split_ind]
    test_labels = labels[split_ind:]

    return (
        train_excitations,
        np.array(train_labels),
        test_excitations,
        np.array(test_labels),
    )


@pytest.mark.parametrize(("solver", "maxiter"), [("Adam", 20)])
def test_qcnn(solver: str, maxiter: int):
    nqubit = 8
    random_seed = 0
    circuit = create_qcnn_ansatz(nqubit, random_seed)

    num_class = 2
    qcl = QNNClassifier(circuit, num_class, solver)

    x_train, y_train, x_test, y_test = generate_data(nqubit)
    qcl.fit(x_train, y_train, maxiter)
    y_pred = qcl.predict(x_test)
    score = f1_score(y_test, y_pred, average="weighted")
    # print("score:", score)
    assert score > 0.9
    return x_test, y_test, y_pred


def main():
    x_test, y_test, y_pred = test_qcnn("Adam", 20)
    plt.plot(x_test, y_test, "o", label="Test")
    plt.plot(x_test, y_pred, "o", label="Prediction")
    plt.legend()
    plt.show()
    # plt.savefig("qcnn.png")


if __name__ == "__main__":
    main()
