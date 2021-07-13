from __future__ import annotations
from functools import reduce
from skqulacs.qnn.qnnbase import (
    QNN,
    _make_fullgate,
    _create_time_evol_gate,
    _get_x_scale_param,
    _min_max_scaling,
    _softmax,
    make_hamiltonian,
)
from qulacs import QuantumState, QuantumCircuit, ParametricQuantumCircuit, Observable
from qulacs.gate import X, Z, DenseMatrix, RX, RY, RZ
from scipy.optimize import minimize
from typing import List, Literal, Optional, Tuple
import numpy as np
from numpy.random import RandomState
from sklearn.metrics import mean_squared_error

# 基本ゲート
I_mat = np.eye(2, dtype=complex)
X_mat = X(0).get_matrix()
Z_mat = Z(0).get_matrix()


class QNNRegressor(QNN):
    """quantum circuit learningを用いて分類問題を解く"""

    def __init__(
        self,
        n_qubit: int,
        circuit_depth: int,
        seed: int = 0,
        time_step: float = 0.7,
        solver: Literal["BFGS", "Nelder-Mead"] = "Nelder-Mead",
        circuit_arch: Literal["default"] = "default",
        n_shot: int = np.inf,
        cost: Literal["mse"] = "mse",
    ) -> None:
        """
        :param nqubit: qubitの数。必要とする出力の次元数よりも多い必要がある
        :param c_depth: circuitの深さ

        """
        self.n_qubit = n_qubit
        self.circuit_depth = circuit_depth
        self.time_step = time_step
        self.solver = solver
        self.circuit_arch = circuit_arch

        self.n_shot = n_shot
        self.cost = cost

        self.scale_x_param = []
        self.scale_y_param = []  # yのスケーリングのパラメータ

        self.obss = []
        self.random_state = RandomState(seed)
        self.seed = seed

        self.output_gate = self._init_output_gate()  # U_out
        self.n_outputs = 0

    def fit(
        self, x_train, y_train, maxiter: Optional[int] = None
    ) -> Tuple[float, np.ndarray]:
        """
        :param x_list: fitしたいデータのxのリスト
        :param y_list: fitしたいデータのyのリスト
        :param maxiter: scipy.optimize.minimizeのイテレーション回数
        :return: 学習後のロス関数の値
        :return: 学習後のパラメータthetaの値
        """

        # 乱数でU_outを作成
        self._init_output_gate()

        self.scale_x_param = _get_x_scale_param(x_train)
        self.scale_y_param = self.get_y_scale_param(y_train)
        # x_trainからscaleのparamを取得
        # regreはyもscaleさせる
        x_scaled = _min_max_scaling(x_train, self.scale_x_param)
        y_scaled = self.do_y_scale(y_train)

        self.obss = [Observable(self.n_qubit) for _ in range(self.n_qubit)]
        if y_train.ndim == 2:
            self.n_outputs = len(y_train[0])
        else:
            self.n_outputs = 1

        for i in range(self.n_qubit):
            self.obss[i].add_operator(1.0, f"Z {i}")  # Z0, Z1, Z2をオブザーバブルとして設定

        theta_init = self._get_output_gate_parameters()
        print(theta_init)
        result = minimize(
            self.cost_func,
            theta_init,
            args=(x_train, y_train),
            method=self.solver,
            # jac=self._cost_func_grad,
            options={"maxiter": maxiter},
        )
        print(self._get_output_gate_parameters())
        loss = result.fun
        theta_opt = result.x
        return loss, theta_opt

    def predict(self, x_test: List[List[float]]):
        # x_test = array-like of of shape (n_samples, n_features)
        x_scaled = _min_max_scaling(x_test, self.scale_x_param)
        y_pred = self.rev_y_scale(self._predict_inner(x_scaled))
        return y_pred

    def _predict_inner(self, x_list):
        # 入力xに関して、量子回路を通した生のデータを表示
        res = []
        # 出力状態計算 & 観測
        for x in x_list:
            state = self._get_input_state(x)
            # U_outで状態を更新
            self.output_gate.update_quantum_state(state)
            # モデルの出力
            r = [
                self.obss[i].get_expectation_value(state) for i in range(self.n_qubit)
            ]  # 出力多次元ver
            res.append(r)
        return np.array(res)

    def _get_input_state(self, x):
        # xはn_features次元のnp.array()
        state = QuantumState(self.n_qubit)

        for i in range(self.n_qubit):
            xa = x[i % len(x)]
            xa = min(1, max(-1, xa))
            RY(i, np.arcsin(xa)).update_quantum_state(state)
            RZ(i, np.arccos(xa * xa)).update_quantum_state(state)
        return state

    def _init_output_gate(self):
        u_out = ParametricQuantumCircuit(self.n_qubit)
        time_evol_gate = _create_time_evol_gate(
            self.n_qubit, time_step=self.time_step, random_state=self.random_state
        )

        for _ in range(self.circuit_depth):

            u_out.add_gate(time_evol_gate)
            for i in range(self.n_qubit):
                angle = 2.0 * np.pi * self.random_state.rand() - np.pi
                u_out.add_parametric_RX_gate(i, angle)
                angle = 2.0 * np.pi * self.random_state.rand() - np.pi
                u_out.add_parametric_RZ_gate(i, angle)
                angle = 2.0 * np.pi * self.random_state.rand() - np.pi
                u_out.add_parametric_RX_gate(i, angle)
        return u_out

    def _update_output_gate(self, theta):
        """U_outをパラメータθで更新"""
        parameter_count = self.output_gate.get_parameter_count()
        for i in range(parameter_count):
            self.output_gate.set_parameter(i, theta[i])

    def _get_output_gate_parameters(self):
        """U_outのパラメータθを取得"""
        parameter_count = self.output_gate.get_parameter_count()
        theta = [
            self.output_gate.get_parameter(index) for index in range(parameter_count)
        ]
        return np.array(theta)

    def cost_func(self, theta, x_train, y_train):
        # 生のデータを入れる
        if self.cost == "mse":
            # mse (default)
            self._update_output_gate(theta)
            y_pred = self.predict(x_train)
            costa = mean_squared_error(y_pred, y_train)
            cost = ((y_pred - y_train) ** 2).mean()
            print(costa)
            return costa
        else:
            raise NotImplementedError(
                f"Cost function {self.cost} is not implemented yet."
            )

    def get_y_scale_param(self, y):
        # 複数入力がある場合に対応したい
        saisyo = np.min(y, axis=0)
        saidai = np.max(y, axis=0)
        sa = (saidai - saisyo) / 2 * 1.7

        return [saisyo, saidai, sa]

    def do_y_scale(self, y):
        # yを[-1,1]の範囲に収める
        # print([((ya - self.scale_y_param[0]) / self.scale_y_param[2]) - 1 for ya in y])
        return [((ya - self.scale_y_param[0]) / self.scale_y_param[2]) - 1 for ya in y]

    def rev_y_scale(self, y_inr):
        # y_inrに含まれる数を、　self.scale_paramを用いて復元する
        return [
            (
                ((ya[0 : self.n_outputs] + 1) * self.scale_y_param[2])
                + self.scale_y_param[0]
            )
            for ya in y_inr
        ]

    """
    # for BFGS
    def _cost_func_grad(self, theta, x_train):
        x_scaled=_min_max_scaling(x_test)
        y_minus_t = self._predict_inner(theta, x_scaled) - self.y_list
        B_grad_list = self._b_grad(theta, x_scaled)
        grad = [np.sum(y_minus_t * B_gr) for B_gr in B_grad_list]
        return np.array(grad)

    # for BFGS
    def _b_grad(self, theta, x_train):
        # dB/dθのリストを返す
        theta_plus = [
            theta.copy() + np.eye(len(theta))[i] * np.pi / 2.0
            for i in range(len(theta))
        ]
        theta_minus = [
            theta.copy() - np.eye(len(theta))[i] * np.pi / 2.0
            for i in range(len(theta))
        ]

        grad = [
            (
                self._predict_inner(theta_plus[i], x_train)
                - self._predict_inner(theta_minus[i], x_train)
            )
            / 2.0
            for i in range(len(theta))
        ]
        return np.array(grad)
    """
