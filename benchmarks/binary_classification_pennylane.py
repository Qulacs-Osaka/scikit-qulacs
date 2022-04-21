from typing import Tuple

import pandas as pd
import pennylane as qml
from numpy.random import Generator
from pennylane import NesterovMomentumOptimizer
from pennylane import numpy as np
from sklearn import datasets
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

dev = qml.device("default.qubit", wires=2)


def get_angles(x):
    beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
    beta1 = 2 * np.arcsin(np.sqrt(x[3] ** 2) / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
    beta2 = 2 * np.arcsin(
        np.sqrt(x[2] ** 2 + x[3] ** 2)
        / np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2)
    )

    return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])


def prepare_state(a):
    qml.RY(a[0], wires=0)

    qml.CNOT(wires=[0, 1])
    qml.RY(a[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[2], wires=1)

    qml.PauliX(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[3], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[4], wires=1)
    qml.PauliX(wires=0)


def layer(W):
    qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
    qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
    qml.CNOT(wires=[0, 1])


@qml.qnode(dev)
def circuit(weights, angles):
    prepare_state(angles)

    for W in weights:
        layer(W)

    return qml.expval(qml.PauliZ(0))


def variational_classifier(weights, bias, angles):
    return circuit(weights, angles) + bias


def square_loss(labels, predictions):
    loss = 0.0
    for label, prediction in zip(labels, predictions):
        loss += (label - prediction) ** 2

    loss = loss / len(labels)
    return loss


def cost(weights, bias, features, labels):
    assert len(features) == len(labels)
    predictions = [variational_classifier(weights, bias, f) for f in features]
    return square_loss(labels, predictions)


def accuracy(labels, predictions):
    assert len(labels) == len(predictions)
    n_corrects = 0
    for label, prediction in zip(labels, predictions):
        if abs(label - prediction) < 1e-5:
            n_corrects = n_corrects + 1
    n_corrects = n_corrects / len(labels)

    return n_corrects


def load_iris_pennylane() -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    x = df.loc[:, ["petal length (cm)", "petal width (cm)"]]
    x = x.to_numpy()

    # Each elements in `iris.target` is 0, 1 or 2.
    # Exclude labels 2 for binary classification.
    index = iris.target != 2
    x = x[index]
    y = np.array(iris.target[index], requires_grad=False)
    # In this learning process, a train label should be -1 or 1; 0 -> -1, 1 -> 1
    y = y * 2 - 1

    padding = 0.3 * np.ones((len(x), 1))
    x_pad = np.c_[np.c_[x, padding], np.zeros((len(x), 1))]

    # normalize each input
    normalization = np.sqrt(np.sum(x_pad ** 2, -1))
    x_norm = (x_pad.T / normalization).T

    # angles for state preparation are new features
    features = np.array([get_angles(x) for x in x_norm], requires_grad=False)

    x_train, x_test, y_train, y_test = train_test_split(
        features, y, test_size=0.25, random_state=0
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_test, y_test, test_size=0.5, random_state=0
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


def train(
    x_train: np.ndarray,
    x_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    rng: Generator,
):
    num_qubits = 2
    num_layers = 6

    opt = NesterovMomentumOptimizer(0.01)
    batch_size = 5

    # train the variational classifier
    weights = 0.01 * rng.random((num_layers, num_qubits, 3))
    # Convert for autograd
    weights = np.array(weights, requires_grad=True)
    bias = np.array(0.0, requires_grad=True)
    best_weights = weights
    best_bias = bias
    best_acc_val = 0.0
    n_epoch = 20
    for epoch in range(n_epoch):
        # Update the weights by one optimizer step
        batch_index = rng.integers(0, len(x_train), (batch_size,))
        x_train_batch = x_train[batch_index]
        y_train_batch = y_train[batch_index]
        weights, bias, _, _ = opt.step(
            cost, weights, bias, x_train_batch, y_train_batch
        )

        # Compute predictions on train and validation set
        predictions_val = [
            np.sign(variational_classifier(weights, bias, x)) for x in x_val
        ]

        # Compute accuracy on validation set
        acc_val = accuracy(y_val, predictions_val)
        if acc_val >= best_acc_val:
            best_weights, best_bias = weights, bias

    return best_weights, best_bias


def binary_classification_pennylane(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    rng: Generator,
) -> None:
    weights, bias = train(x_train, x_val, y_train, y_val, rng)
    y_pred = [np.sign(variational_classifier(weights, bias, x)) for x in x_test]
    f1_score(y_test, y_pred) > 0.95
