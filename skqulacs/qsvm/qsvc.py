import numpy as np
from qulacs_osaka.state import inner_product
from sklearn import svm

from skqulacs.circuit import LearningCircuit


class QSVC:
    def __init__(self, circuit: LearningCircuit) -> None:
        self.regr = svm.SVC(kernel="precomputed")
        self.circuit = circuit
        self.data_states = []
        self.n_qubit = 0

    def fit(self, x, y):
        # print(x)
        self.n_qubit = len(x[0])
        kar = np.zeros((len(x), len(x)))  # サンプル数の二乗の情報量　距離を入れる
        # xとyのカーネルを計算する
        # そのために、UΦxを計算する
        for i in range(len(x)):
            self.data_states.append(self.circuit.run(x[i]))

        for i in range(len(x)):
            # print(self.data_states[i])
            for j in range(len(x)):
                kar[i][j] = (
                    abs(inner_product(self.data_states[i], self.data_states[j])) ** 2
                )

        # print(kar)
        self.regr.fit(kar, y)

    def predict(self, xs):
        kar = np.zeros((len(xs), len(self.data_states)))  # サンプル数の二乗の情報量　距離を入れる
        for i in range(len(xs)):
            x_qc = self.circuit.run(xs[i])
            for j in range(len(self.data_states)):
                kar[i][j] = abs(inner_product(x_qc, self.data_states[j])) ** 2
        # print(kar)
        return self.regr.predict(kar)
