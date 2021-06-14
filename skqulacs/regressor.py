from __future__ import annotations
from functools import reduce
from qulacs import QuantumState, QuantumCircuit, ParametricQuantumCircuit, Observable
from qulacs.gate import X, Z, DenseMatrix
from scipy.optimize import minimize
from typing import Literal, Tuple
import numpy as np


# 基本ゲート
I_mat = np.eye(2, dtype=complex)
X_mat = X(0).get_matrix()
Z_mat = Z(0).get_matrix()


# fullsizeのgateをつくる関数.
def _make_fullgate(list_SiteAndOperator, nqubit):
    """
    list_SiteAndOperator = [ [i_0, O_0], [i_1, O_1], ...] を受け取り,
    関係ないqubitにIdentityを挿入して
    I(0) * ... * O_0(i_0) * ... * O_1(i_1) ...
    という(2**nqubit, 2**nqubit)行列をつくる.
    """
    list_Site = [SiteAndOperator[0] for SiteAndOperator in list_SiteAndOperator]
    list_SingleGates = []  # 1-qubit gateを並べてnp.kronでreduceする
    cnt = 0
    for i in range(nqubit):
        if i in list_Site:
            list_SingleGates.append(list_SiteAndOperator[cnt][1])
            cnt += 1
        else:  # 何もないsiteはidentity
            list_SingleGates.append(I_mat)

    return reduce(np.kron, list_SingleGates)


def _create_time_evol_gate(nqubit, time_step=0.77):
    """ランダム磁場・ランダム結合イジングハミルトニアンをつくって時間発展演算子をつくる
    :param time_step: ランダムハミルトニアンによる時間発展の経過時間
    :return  qulacsのゲートオブジェクト
    """
    ham = np.zeros((2 ** nqubit, 2 ** nqubit), dtype=complex)
    for i in range(nqubit):  # i runs 0 to nqubit-1
        Jx = -1.0 + 2.0 * np.random.rand()  # -1~1の乱数
        ham += Jx * _make_fullgate([[i, X_mat]], nqubit)
        for j in range(i + 1, nqubit):
            J_ij = -1.0 + 2.0 * np.random.rand()
            ham += J_ij * _make_fullgate([[i, Z_mat], [j, Z_mat]], nqubit)

    # 対角化して時間発展演算子をつくる. H*P = P*D <-> H = P*D*P^dagger
    diag, eigen_vecs = np.linalg.eigh(ham)
    time_evol_op = np.dot(
        np.dot(eigen_vecs, np.diag(np.exp(-1j * time_step * diag))), eigen_vecs.T.conj()
    )  # e^-iHT

    # qulacsのゲートに変換
    time_evol_gate = DenseMatrix(range(nqubit), time_evol_op)
    return time_evol_gate


def _make_hamiltonian(n_qubit):
    ham = np.zeros((2 ** n_qubit, 2 ** n_qubit), dtype=complex)
    X_mat = X(0).get_matrix()
    Z_mat = Z(0).get_matrix()
    for i in range(n_qubit):
        Jx = -1.0 + 2.0 * np.random.rand()
        ham += Jx * _make_fullgate([[i, X_mat]], n_qubit)
        for j in range(i + 1, n_qubit):
            J_ij = -1.0 + 2.0 * np.random.rand()
            ham += J_ij * _make_fullgate([[i, Z_mat], [j, Z_mat]], n_qubit)
    return ham


def min_max_scaling(x, axis=None):
    """[-1, 1]の範囲に規格化"""
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x - min) / (max - min)
    result = 2.0 * result - 1.0
    return result


def softmax(x):
    """softmax function
    :param x: ndarray
    """
    exp_x = np.exp(x)
    y = exp_x / np.sum(np.exp(x))
    return y


class QNNRegressor:
    """Solve regression tasks with Quantum Neural Network.

    Args:
        n_qubit: The number of qubits.
        circuit_depth: Depth of parametric gate.
        time_step: Elapsed time in time evolution of Hamiltonian.
        solver: Kind of optimization solver. Methods in scipy.optimize.minimize are supported.
        circuit_arch: Form of quantum circuit.
        observable: String representation of observable operators.
        n_shots: The number of measurements to compute expectation.
        cost: Kind of cost function.
    Attributes:
        observable: Observable.
    """

    OBSERVABLE_COEF: float = 2.0

    def __init__(
        self,
        n_qubit: int,
        circuit_depth: int,
        time_step: float,
        solver: Literal["BFGS", "Nelder-Mead"] = "Nelder-Mead",
        circuit_arch: Literal["default"] = "default",
        observable: str = "Z 0",
        n_shot: int = np.inf,
        cost: Literal["mse"] = "mse",
    ) -> None:
        self.n_qubit = n_qubit
        self.circuit_depth = circuit_depth
        self.time_step = time_step
        self.solver = solver
        self.circuit_arch = circuit_arch
        self.observable = Observable(n_qubit)
        self.observable.add_operator(self.OBSERVABLE_COEF, observable)
        self.n_shot = n_shot
        self.cost = cost
        self.u_out = self._u_output()
        self.obs = Observable(self.n_qubit)

    def fit(self, x, y) -> Tuple[float, np.ndarray]:
        """Fit model.

        Args:
            x: train data of x.
            y: train data of y.

        Returns:
            loss: Loss of optimized model.
            theta_opt: Parameter of optimized model.
        """
        self.obs.add_operator(2.0, "Z 0")
        parameter_count = self.u_out.get_parameter_count()
        theta_init_a = [self.u_out.get_parameter(ind) for ind in range(parameter_count)]
        theta_init = theta_init_a.copy()
        result = minimize(
            QNNRegressor._cost_func,
            theta_init,
            args=(self, x, y),
            method=self.solver,
        )

        theta_opt = result.x
        self._update_u_out(theta_opt)
        loss = result.fun
        return loss, theta_opt

    def predict(self, x) -> float:
        """Predict outcome of given x."""
        state = QuantumState(self.n_qubit)
        state.set_zero_state()
        self._u_input(x).update_quantum_state(state)
        self.u_out.update_quantum_state(state)
        return self.obs.get_expectation_value(state)

    @staticmethod
    def _cost_func(theta, model: QNNRegressor, x_train, y_train):
        model._update_u_out(theta)
        y_predict = [model.predict(x) for x in x_train]
        if model.cost == "mse":
            return ((y_predict - y_train) ** 2).mean()
        else:
            raise NotImplementedError(
                f"Cost function {model.cost} is not implemented yet."
            )

    def _u_input(self, x):
        u_in = QuantumCircuit(self.n_qubit)
        angle_y = np.arcsin(x)
        angle_z = np.arccos(x ** 2)
        for i in range(self.n_qubit):
            u_in.add_RY_gate(i, angle_y)
            u_in.add_RZ_gate(i, angle_z)
        return u_in

    def _u_output(self):
        time_evol_gate = _create_time_evol_gate(self.n_qubit, self.time_step)
        u_out = ParametricQuantumCircuit(self.n_qubit)
        for _ in range(self.circuit_depth):
            u_out.add_gate(time_evol_gate)
            for i in range(self.n_qubit):
                angle = 2.0 * np.pi * np.random.rand()
                u_out.add_parametric_RX_gate(i, angle)
                angle = 2.0 * np.pi * np.random.rand()
                u_out.add_parametric_RZ_gate(i, angle)
                angle = 2.0 * np.pi * np.random.rand()
                u_out.add_parametric_RX_gate(i, angle)
        return u_out

    def _time_evol_gate(self):
        hamiltonian = _make_hamiltonian(self.n_qubit)
        diag, eigen_vecs = np.linalg.eigh(hamiltonian)
        time_evol_op = np.dot(
            np.dot(eigen_vecs, np.diag(np.exp(-1j * self.time_step * diag))),
            eigen_vecs.T.conj(),
        )
        return DenseMatrix(range(self.n_qubit), time_evol_op)

    def _update_u_out(self, theta):
        param_count = self.u_out.get_parameter_count()
        for i in range(param_count):
            self.u_out.set_parameter(i, theta[i])
