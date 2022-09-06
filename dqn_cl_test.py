import csv
from typing import Tuple

import numpy as np
from qulacs import Observable
from qulacs.gate import CZ
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from skqulacs.circuit.circuit import LearningCircuit
from skqulacs.circuit.pre_defined import (
    create_dqn_cl,
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


# This script aims to reproduce Ⅳ.B Binary classification in https://arxiv.org/pdf/2112.15002.pdf.

# This is the same as the number of qubits here.
n_features = 13
locality = 2

for depth in range(3, 10):
    # circuit = create_dqn_cl(n_features, c_depth=depth, s_qubit=2)
    circuit = create_farhi_neven_ansatz(n_features, c_depth=3)

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

    classifier.fit(np.array(x_train), np.array(y_train), 9)
    y_pred = classifier.predict(np.array(x_test))
    print("depth=", depth)
    print(classification_report(y_test, y_pred, labels=[0, 1]))

"""
結果
accuracy　単純比較
  3    4    5    6    7    8    9  (depth)
0.74 0.86 0.94 0.63 0.91 0.80 0.83 create_dqn_cl (新)
0.88 0.58 0.54 0.37 0.89 0.62 0.86 create_farhi_neven_ansatz(旧)

結論
かなり誤差というか、不安定というか、　うまくいくときといかないときがある
だけど、　create_dqn_cl のほうが全体的によさそうだというこちがわかる
計算は3分ぐらいかかった  正確に調べたければ複数回回す必要があるが、時間がかかりそう


dqn
depth= 3
              precision    recall  f1-score   support

           0       0.86      0.56      0.68        32
           1       0.68      0.91      0.78        33

    accuracy                           0.74        65
   macro avg       0.77      0.74      0.73        65
weighted avg       0.77      0.74      0.73        65

depth= 4
              precision    recall  f1-score   support

           0       0.86      0.83      0.84        29
           1       0.86      0.89      0.88        36

    accuracy                           0.86        65
   macro avg       0.86      0.86      0.86        65
weighted avg       0.86      0.86      0.86        65

depth= 5
              precision    recall  f1-score   support

           0       0.93      0.93      0.93        27
           1       0.95      0.95      0.95        38

    accuracy                           0.94        65
   macro avg       0.94      0.94      0.94        65
weighted avg       0.94      0.94      0.94        65

depth= 6
              precision    recall  f1-score   support

           0       0.67      0.40      0.50        30
           1       0.62      0.83      0.71        35

    accuracy                           0.63        65
   macro avg       0.64      0.61      0.60        65
weighted avg       0.64      0.63      0.61        65

depth= 7
              precision    recall  f1-score   support

           0       0.90      0.90      0.90        29
           1       0.92      0.92      0.92        36

    accuracy                           0.91        65
   macro avg       0.91      0.91      0.91        65
weighted avg       0.91      0.91      0.91        65

depth= 8
              precision    recall  f1-score   support

           0       0.73      0.86      0.79        28
           1       0.88      0.76      0.81        37

    accuracy                           0.80        65
   macro avg       0.80      0.81      0.80        65
weighted avg       0.81      0.80      0.80        65

depth= 9
              precision    recall  f1-score   support

           0       0.71      0.87      0.78        23
           1       0.92      0.81      0.86        42

    accuracy                           0.83        65
   macro avg       0.82      0.84      0.82        65
weighted avg       0.85      0.83      0.83        65


fari
depth= 3
              precision    recall  f1-score   support

           0       0.79      0.96      0.87        28
           1       0.97      0.81      0.88        37

    accuracy                           0.88        65
   macro avg       0.88      0.89      0.88        65
weighted avg       0.89      0.88      0.88        65

depth= 4
              precision    recall  f1-score   support

           0       0.58      0.24      0.34        29
           1       0.58      0.86      0.70        36

    accuracy                           0.58        65
   macro avg       0.58      0.55      0.52        65
weighted avg       0.58      0.58      0.54        65

depth= 5
              precision    recall  f1-score   support

           0       0.46      0.39      0.42        28
           1       0.59      0.65      0.62        37

    accuracy                           0.54        65
   macro avg       0.52      0.52      0.52        65
weighted avg       0.53      0.54      0.53        65

depth= 6
              precision    recall  f1-score   support

           0       0.38      0.92      0.54        26
           1       0.00      0.00      0.00        39

    accuracy                           0.37        65
   macro avg       0.19      0.46      0.27        65
weighted avg       0.15      0.37      0.22        65

depth= 7
              precision    recall  f1-score   support

           0       0.78      1.00      0.88        25
           1       1.00      0.82      0.90        40

    accuracy                           0.89        65
   macro avg       0.89      0.91      0.89        65
weighted avg       0.92      0.89      0.89        65

depth= 8
              precision    recall  f1-score   support

           0       0.57      0.74      0.65        31
           1       0.68      0.50      0.58        34

    accuracy                           0.62        65
   macro avg       0.63      0.62      0.61        65
weighted avg       0.63      0.62      0.61        65

depth= 9
              precision    recall  f1-score   support

           0       0.77      0.92      0.84        25
           1       0.94      0.82      0.88        40

    accuracy                           0.86        65
   macro avg       0.85      0.87      0.86        65
weighted avg       0.88      0.86      0.86        65

-------------
結果
accuracy 単純比較
  3    4    5    6    7    8    9  (depth)
0.74 0.86 0.94 0.63 0.91 0.80 0.83 create_dqn_cl (新)
0.88 0.58 0.54 0.37 0.89 0.62 0.86 create_farhi_neven_ansatz(旧)
"""
