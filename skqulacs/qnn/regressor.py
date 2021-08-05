from __future__ import annotations
from skqulacs.qnn.qnnbase import (
    QNN,
    _get_x_scale_param,
    _min_max_scaling,
)
from qulacs import Observable
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from skqulacs.circuit import LearningCircuit
from skqulacs.typing import Literal
from typing import List, Optional, Tuple
import numpy as np


class QNNRegressor(QNN):
    """quantum circuit learningを用いて分類問題を解く"""

    def __init__(
        self,
        n_qubit: int,
        circuit: LearningCircuit,
        solver: Literal["Adam", "BFGS", "Nelder-Mead"] = "Nelder-Mead",
        cost: Literal["mse"] = "mse",
    ) -> None:
        """
        :param nqubit: qubitの数。必要とする出力の次元数よりも多い必要がある
        :param c_depth: circuitの深さ

        """
        self.n_qubit = n_qubit
        self.circuit = circuit
        self.solver = solver
        self.cost = cost

        self.scale_x_param = []
        self.scale_y_param = []  # yのスケーリングのパラメータ

        self.observables = [Observable(n_qubit) for _ in range(n_qubit)]
        for i in range(n_qubit):
            self.observables[i].add_operator(1.0, f"Z {i}")

    def fit(
        self,
        x_train: List[List[float]],
        y_train: List[float],
        maxiter: Optional[int] = None,
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
        if self.solver == "Nelder-Mead":
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
        elif self.solver == "BFGS":
            result = minimize(
                self.cost_func,
                theta_init,
                args=(x_train, y_train),
                method=self.solver,
                jac=self._cost_func_grad,
                options={"maxiter": maxiter},
            )
            loss = result.fun
            theta_opt = result.x
        elif self.solver == "Adam":
            pr_A = 0.25
            pr_Bi = 0.6
            pr_Bt = 0.99
            pr_ips = 0.0000001
            # ここまでがハイパーパラメータ
            Bix = 0
            Btx = 0

            moment = np.zeros(len(theta_init))
            vel = 0
            theta_now = theta_init
            maxiter *= len(x_train)
            for iter in range(0, maxiter, 5):
                grad = self._cost_func_grad(
                    theta_now,
                    x_train[iter % len(x_train) : iter % len(x_train) + 5],
                    y_train[iter % len(y_train) : iter % len(y_train) + 5],
                )
                moment = moment * pr_Bi + (1 - pr_Bi) * grad
                vel = vel * pr_Bt + (1 - pr_Bt) * np.dot(grad, grad)
                Bix = Bix * pr_Bi + (1 - pr_Bi)
                Btx = Btx * pr_Bt + (1 - pr_Bt)
                theta_now -= pr_A / (((vel / Btx) ** 0.5) + pr_ips) * (moment / Bix)
                if iter % len(x_train) < 5:
                    self.cost_func(theta_now, x_train, y_train)

            loss = self.cost_func(theta_now, x_train, y_train)
            theta_opt = theta_now
        else:
            raise NotImplementedError

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

    def _cost_func_grad(self, theta, x_train, y_train):
        self.circuit.update_parameters(theta)
        x_scaled = _min_max_scaling(x_train, self.scale_x_param)
        y_scaled = self._do_y_scale(y_train)
        mto = self._predict_inner(x_scaled).copy()
        bbb = np.zeros((len(x_train), self.n_qubit))
        for h in range(len(x_train)):
            if self.n_outputs == 2:
                for i in range(self.n_outputs):
                    bbb[h][i] = (-y_scaled[h][i] + mto[h][i]) / self.n_outputs
            else:
                bbb[h] = (-y_scaled[h] + mto[h]) / self.n_outputs

        theta_plus = [
            theta.copy() + (np.eye(len(theta))[i] / 20.0) for i in range(len(theta))
        ]
        theta_minus = [
            theta.copy() - (np.eye(len(theta))[i] / 20.0) for i in range(len(theta))
        ]

        grad = np.zeros(len(theta))
        for i in range(len(theta)):
            self.circuit.update_parameters(theta_plus[i])
            aaa_f = self._predict_inner(x_scaled)
            self.circuit.update_parameters(theta_minus[i])
            aaa_m = self._predict_inner(x_scaled)
            for j in range(len(x_train)):
                grad[i] += np.dot(aaa_f[j] - aaa_m[j], bbb[j]) * 10.0

        self.circuit.update_parameters(theta)
        grad /= len(x_train)
        return grad
