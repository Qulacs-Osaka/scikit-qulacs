import csv
from typing import Tuple
import numpy as np
from qulacs import Observable
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from skqulacs.circuit.circuit import LearningCircuit
from qulacs.gate import CZ

from skqulacs.qnn.classifier import QNNClassifier
from skqulacs.qnn.solver import Adam
from skqulacs.circuit.pre_defined import create_dqn_cl



# Use wine dataset retrieved from: https://archive-beta.ics.uci.edu/ml/datasets/wine
def load_dataset(file_path: str, ignore_kind: int, test_ratio: float) -> Tuple[np.array, np.array, np.array, np.array]:
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

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio, shuffle=True)

    return x_train, x_test, y_train, y_test


# This script aims to reproduce Ⅳ.B Binary classification in https://arxiv.org/pdf/2112.15002.pdf.

# This is the same as the number of qubits here.
n_features = 13
locality = 2
circuit = create_dqn_cl(n_features,  c_depth=2)

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
    y_train[i]-=1 
for i in range(len(y_test)):
    y_test[i]-=1 
classifier.fit(np.array(x_train), np.array(y_train), 100)
y_pred = classifier.predict(np.array(x_test))



print(classification_report(y_test, y_pred, labels=[0, 1]))

print(x_train)
print(y_train)

print(y_test)
print(y_pred)
