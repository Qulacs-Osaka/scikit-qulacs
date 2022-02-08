import pandas as pd
import pennylane as qml
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


def statepreparation(a):
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
    statepreparation(angles)

    for W in weights:
        layer(W)

    return qml.expval(qml.PauliZ(0))


def variational_classifier(weights, bias, angles):
    return circuit(weights, angles) + bias


def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss += (l - p) ** 2

    loss = loss / len(labels)
    return loss


def cost(weights, bias, features, labels):
    assert len(features) == len(labels)
    predictions = [variational_classifier(weights, bias, f) for f in features]
    return square_loss(labels, predictions)


def accuracy(labels, predictions):
    assert len(labels) == len(predictions)
    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)

    return loss


iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
X = df.loc[:, ["petal length (cm)", "petal width (cm)"]]
Y = iris.target.astype(np.float32)
X = X.to_numpy()

index = Y != 2
X = X[index]
Y = Y[index] * 2 - 1

padding = 0.3 * np.ones((len(X), 1))
X_pad = np.c_[np.c_[X, padding], np.zeros((len(X), 1))]

# normalize each input
normalization = np.sqrt(np.sum(X_pad ** 2, -1))
X_norm = (X_pad.T / normalization).T

# angles for state preparation are new features
features = np.array([get_angles(x) for x in X_norm], requires_grad=True)

X_train, X_test, Y_train, Y_test = train_test_split(
    features, Y, test_size=0.25, random_state=0
)

X_val, X_test, Y_val, Y_test = train_test_split(
    X_test, Y_test, test_size=0.5, random_state=0
)


def train(X_train, Y_train):
    num_qubits = 2
    num_layers = 5

    opt = NesterovMomentumOptimizer(0.01)
    batch_size = 5

    # train the variational classifier
    weights = 0.01 * np.random.randn(num_layers, num_qubits, 3, requires_grad=True)
    bias = np.array(0.0, requires_grad=True)
    best_weights = weights
    best_bias = bias
    best_acc_val = 0.0
    n_epoch = 20
    for epoch in range(n_epoch):
        # Update the weights by one optimizer step
        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_train_batch = X_train[batch_index]
        Y_train_batch = Y_train[batch_index]
        weights, bias, _, _ = opt.step(
            cost, weights, bias, X_train_batch, Y_train_batch
        )

        # Compute predictions on train and validation set
        predictions_train = [
            np.sign(variational_classifier(weights, bias, x)) for x in X_train
        ]
        predictions_val = [
            np.sign(variational_classifier(weights, bias, x)) for x in X_val
        ]

        # Compute accuracy on train and validation set
        acc_train = accuracy(Y_train, predictions_train)
        acc_val = accuracy(Y_val, predictions_val)
        if acc_val >= best_acc_val:
            best_weights, best_bias = weights, bias

        print(
            "Epoch: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f} | Acc val: {:0.7f}"
            "".format(epoch + 1, cost(weights, bias, features, Y), acc_train, acc_val)
        )

    return best_weights, best_bias


def binary_classification_pennylane():
    weights, bias = train(X_train, Y_train)
    Y_pred = [np.sign(variational_classifier(weights, bias, x)) for x in X_test]
    return f1_score(Y_test, Y_pred)
