import numpy as np
from numpy.typing import NDArray
from qulacs.state import inner_product
from sklearn import svm

from skqulacs.circuit import LearningCircuit


class QSVR:
    """class to solve regression problems with support vector regressor with a quantum kernel"""

    def __init__(self, circuit: LearningCircuit) -> None:
        """
        :param circuit: circuit to generate quantum feature
        """
        self.svr = svm.SVR(kernel="precomputed")
        self.circuit = circuit
        self.data_states = []
        self.n_qubit = 0

    def fit(self, x: NDArray[np.float_], y: NDArray[np.int_]):
        """
        train the machine.
        :param x: training inputs
        :param y: training teacher values
        """
        self.n_qubit = len(x[0])
        kar = np.zeros((len(x), len(x)))
        # Compute UΦx to get kernel of `x` and `y`.
        for i in range(len(x)):
            self.data_states.append(self.circuit.run(x[i]))

        for i in range(len(x)):
            for j in range(len(x)):
                kar[i][j] = (
                    abs(inner_product(self.data_states[i], self.data_states[j])) ** 2
                )

        self.svr.fit(kar, y)

    def predict(self, xs: NDArray[np.float_]):
        """
        predict y values for each of xs
        :param xs: inputs to make predictions
        :return: List[int], predicted values of y
        """
        kar = np.zeros((len(xs), len(self.data_states)))
        for i in range(len(xs)):
            x_qc = self.circuit.run(xs[i])
            for j in range(len(self.data_states)):
                kar[i][j] = abs(inner_product(x_qc, self.data_states[j])) ** 2
        return self.svr.predict(kar)
