from __future__ import annotations
from skqulacs.qnn.qnnbase import (
    QNN,
    _get_x_scale_param,
    _min_max_scaling,
)
from qulacs import Observable
from qulacs.gate import X, Z
from sklearn.metrics import mean_squared_error
from skqulacs.circuit import LearningCircuit
from scipy.optimize import minimize
from typing import List, Literal, Optional, Tuple
from numpy.random import RandomState
import numpy as np

# 基本ゲート
I_mat = np.eye(2, dtype=complex)
X_mat = X(0).get_matrix()
Z_mat = Z(0).get_matrix()


class QNNRegressor(QNN):
    """quantum circuit learningを用いて分類問題を解く"""

    def __init__(
        self,
        n_qubit: int,
        circuit: LearningCircuit,
        seed: int = 0,
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
        self.circuit = circuit
        self.solver = solver
        self.circuit_arch = circuit_arch

        self.n_shot = n_shot
        self.cost = cost

        self.scale_x_param = []
        self.scale_y_param = []  # yのスケーリングのパラメータ

        self.observables = []
        for _ in range(n_qubit):
            observable = Observable(n_qubit)
            for i in range(self.n_qubit):
                observable.add_operator(1.0, f"Z {i}")  # Z0, Z1, Z2をオブザーバブルとして設定
            self.observables.append(observable)
        self.random_state = RandomState(seed)

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
        self.scale_x_param = _get_x_scale_param(x_train)
        self.scale_y_param = self._get_y_scale_param(y_train)
        # x_trainからscaleのparamを取得
        # regreはyもscaleさせる
        # x_scaled = _min_max_scaling(x_train, self.scale_x_param)
        # y_scaled = self._do_y_scale(y_train)

        if y_train.ndim == 2:
            self.n_outputs = len(y_train[0])
        else:
            self.n_outputs = 1

        theta_init = self.circuit.get_parameters()
        result = minimize(
            self.cost_func,
            theta_init,
            args=(x_train, y_train),
            method=self.solver,
            # jac=self._cost_func_grad,
            options={"maxiter": maxiter},
        )
        loss = result.fun
        theta_opt = result.x
        return loss, theta_opt

    def predict(self, x_test: List[List[float]]) -> List[float]:
        """Predict outcome for each input data in `x_test`.

        Arguments:
            x_test: Input data whose shape is (n_samples, n_features).

        Returns:
            y_pred: Predicted outcome.
        """
        x_scaled = _min_max_scaling(x_test, self.scale_x_param)
        y_pred = self._rev_y_scale(self._predict_inner(x_scaled))
        return y_pred

    def _predict_inner(self, x_list: List[List[float]]):
        res = []
        # 出力状態計算 & 観測
        for x in x_list:
            state = self.circuit.run(x)
            # モデルの出力
            r = [
                self.observables[i].get_expectation_value(state)
                for i in range(self.n_qubit)
            ]  # 出力多次元ver
            res.append(r)
        return np.array(res)

    def cost_func(self, theta, x_train, y_train):
        if self.cost == "mse":
            self.circuit.update_parameters(theta)
            y_pred = self.predict(x_train)
            cost = mean_squared_error(y_pred, y_train)
            return cost
        else:
            raise NotImplementedError(
                f"Cost function {self.cost} is not implemented yet."
            )

    def _get_y_scale_param(self, y):
        # 複数入力がある場合に対応したい
        minimum = np.min(y, axis=0)
        maximum = np.max(y, axis=0)
        sa = (maximum - minimum) / 2 * 1.7

        return [minimum, maximum, sa]

    def _do_y_scale(self, y):
        # yを[-1,1]の範囲に収める
        # print([((ya - self.scale_y_param[0]) / self.scale_y_param[2]) - 1 for ya in y])
        return [((ya - self.scale_y_param[0]) / self.scale_y_param[2]) - 1 for ya in y]

    def _rev_y_scale(self, y_inr):
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
