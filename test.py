import csv
from typing import Tuple

import numpy as np
from qulacs import Observable
from qulacs.gate import CZ
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from skqulacs.circuit.circuit import LearningCircuit
from skqulacs.circuit.pre_defined import (
    create_dqn_cl,
    create_dqn_cl_no_cz,
    create_farhi_neven_ansatz,
)
from skqulacs.qnn.classifier import QNNClassifier
from skqulacs.qnn.solver import Adam


# Use wine dataset retrieved from: https://archive-beta.ics.uci.edu/ml/datasets/wine
def load_dataset(
    file_path: str, ignore_kind: int, test_ratio: float
) -> Tuple[np.array, np.array, np.array, np.array]:
    """Load dataset from specified path.

    Args:
        file_path: File path from which data is loaded.
        ignore_kind: The dataset expected to have 3 classes and we need 2 classes to test. So specify here which class to ignore in loading.
    """
    x = []
    y = []
    with open(file_path) as f:
        reader = csv.reader(f)
        for row in reader:
            kind = int(row[0])
            if kind == ignore_kind:
                continue
            y.append(kind)
            x.append([float(feature) for feature in row[1:]])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_ratio, shuffle=True
    )

    return x_train, x_test, y_train, y_test


def create_classifier(n_features, circuit):
    # Observables are hard-coded in QNNClassifier, so overwrite here.
    classifier = QNNClassifier(circuit, 2, Adam())
    classifier.observables = [Observable(n_features) for _ in range(n_features)]
    for i in range(n_features):
        if i < locality:
            classifier.observables[i].add_operator(1.0, f"Z {i}")
        else:
            classifier.observables[i].add_operator(1.0, f"I {i}")
    return classifier


# This script aims to reproduce Ⅳ.B Binary classification in https://arxiv.org/pdf/2112.15002.pdf.

# This is the same as the number of qubits here.
n_features = 13
locality = 2
circuit = create_dqn_cl(n_features, c_depth=4, s_qubit=locality)

# Observables are hard-coded in QNNClassifier, so overwrite here.
classifier = QNNClassifier(circuit, 2, Adam())
classifier.observables = [Observable(n_features) for _ in range(n_features)]
for i in range(n_features):
    if i < locality:
        classifier.observables[i].add_operator(1.0, f"Z {i}")
    else:
        classifier.observables[i].add_operator(1.0, f"I {i}")

x_train, x_test, y_train, y_test = load_dataset("wine.data", 3, 0.5)

for i in range(len(y_train)):
    y_train[i] -= 1
for i in range(len(y_test)):
    y_test[i] -= 1

# classifier.fit(np.array(x_train), np.array(y_train), 15)
# y_pred = classifier.predict(np.array(x_test))
# print(classification_report(y_test, y_pred, labels=[0, 1]))

maxiter = 5
print("CL")
circuit = create_dqn_cl(n_features, 5, locality)
# print(circuit._circuit)
classifier = create_classifier(n_features, circuit)
for i in range(80):
    classifier.fit(np.array(x_train), np.array(y_train), maxiter)
    y_pred = classifier.predict(np.array(x_test))
    score = f1_score(y_test, y_pred, average="weighted")
    print("iteration:", (i + 1) * maxiter, " score:", score)

print("no CL")
circuit = create_dqn_cl_no_cz(n_features, 5)
# print(circuit._circuit)
classifier = create_classifier(n_features, circuit)
for i in range(80):
    classifier.fit(np.array(x_train), np.array(y_train), maxiter)
    y_pred = classifier.predict(np.array(x_test))
    score = f1_score(y_test, y_pred, average="weighted")
    print("iteration:", (i + 1) * maxiter, " score:", score)


"""
create_dqn_cl だとf1-score が 0.89
create_farhi_neven_ansatz だとf1-score が 0.85
だからdqnのほうが良い ？
"""
