from functools import reduce
from qulacs import QuantumState, QuantumCircuit, ParametricQuantumCircuit, Observable
from qulacs.gate import X, Z, DenseMatrix
from scipy.optimize import minimize
from typing import Literal
import numpy as np


class QNNRegressor:
    OBSERVABLE_COEF: float = 2.0

    def __init__(
        self,
        n_qubit: int,
        solver: Literal["bfgs"] = "bfgs",
        circuit_arch: Literal["default"] = "default",
        observable: str = "Z 0",
        n_shot: int = np.inf,
        cost: Literal["mse"] = "mse",
    ) -> None:
        self.n_qubit = n_qubit
        self.solver = solver
        self.circuit_arch = circuit_arch
        self.observable = Observable(n_qubit)
        self.observable.add_operator(self.OBSERVABLE_COEF, observable)
        self.n_shot = n_shot
        self.cost = cost
