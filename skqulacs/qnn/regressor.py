from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from qulacs import Observable
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

from skqulacs.circuit import LearningCircuit
from skqulacs.qnn.qnnbase import QNN, _get_x_scale_param, _min_max_scaling
from skqulacs.typing import Literal


class QNNRegressor(QNN):
    """quantum circuit learningを用いて分類問題を解く"""

    def __init__(
        self,
        circuit: LearningCircuit,
        solver: Literal["Adam", "BFGS", "Nelder-Mead"] = "BFGS",
        cost: Literal["mse"] = "mse",
        do_x_scale: bool = True,
        do_y_scale: bool = True,
        y_norm_range=0.7,
    ) -> None:
        """
        :param circuit: 回路そのもの
        :param solver: 何を使うか Nelderは非推奨
        :param cost: コスト関数 mseしかない。 mseは、正規化した後での二乗和をとる。
        :param do_x_scale xをscaleしますか?
        :param do_x_scale yをscaleしますか?
        :param y_margin  [-y_norm_range,y_norm_range]に正規化.
        """
        self.n_qubit = circuit.n_qubit
        self.circuit = circuit
        self.solver = solver
        self.cost = cost
        self.do_x_scale = do_x_scale
        self.do_y_scale = do_y_scale
        self.y_norm_range = y_norm_range

        self.scale_x_param = []
        self.scale_y_param = []  # yのスケーリングのパラメータ

        self.observables = []

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

        if self.do_x_scale:
            x_scaled = _min_max_scaling(x_train, self.scale_x_param)
        else:
            x_scaled = x_train

        if self.do_y_scale:
            y_scaled = self._do_y_scale(y_train)
        else:
            y_scaled = y_train

        # x_trainからscaleのparamを取得
        # regreはyもscaleさせる
        # x_scaled = _min_max_scaling(x_train, self.scale_x_param)
        # y_scaled = self._do_y_scale(y_scaled)

        if y_train.ndim == 2:
            self.n_outputs = len(y_scaled[0])
        else:
            self.n_outputs = 1

        self.observables = []
        for i in range(self.n_outputs):
            observable = Observable(self.n_qubit)
            # Z0, Z1, Z2をオブザーバブルとして設定
            observable.add_operator(1.0, f"Z {i}")
            self.observables.append(observable)

        theta_init = self.circuit.get_parameters()
        if self.solver == "Nelder-Mead":
            result = minimize(
                self.cost_func,
                theta_init,
                args=(x_scaled, y_scaled),
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
                args=(x_scaled, y_scaled),
                method=self.solver,
                jac=self._cost_func_grad,
                options={"maxiter": maxiter},
            )
            # print(self._cost_func_grad(result.x, x_scaled, y_scaled))
            loss = result.fun
            theta_opt = result.x
        elif self.solver == "Adam":
            pr_A = 0.02
            pr_Bi = 0.8
            pr_Bt = 0.995
            pr_ips = 0.0000001
            # ここまでがハイパーパラメータ
            Bix = 0
            Btx = 0

            moment = np.zeros(len(theta_init))
            vel = 0
            theta_now = theta_init
            maxiter *= len(x_scaled)
            for iter in range(0, maxiter, 5):
                grad = self._cost_func_grad(
                    theta_now,
                    x_scaled[iter % len(x_scaled) : iter % len(x_scaled) + 5],
                    y_scaled[iter % len(y_scaled) : iter % len(y_scaled) + 5],
                )
                moment = moment * pr_Bi + (1 - pr_Bi) * grad
                vel = vel * pr_Bt + (1 - pr_Bt) * np.dot(grad, grad)
                Bix = Bix * pr_Bi + (1 - pr_Bi)
                Btx = Btx * pr_Bt + (1 - pr_Bt)
                theta_now -= pr_A / (((vel / Btx) ** 0.5) + pr_ips) * (moment / Bix)
                if iter % len(x_scaled) < 5:
                    self.cost_func(theta_now, x_scaled, y_scaled)

            loss = self.cost_func(theta_now, x_scaled, y_scaled)
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
        if self.do_x_scale:
            x_scaled = _min_max_scaling(x_test, self.scale_x_param)
        else:
            x_scaled = x_test

        if self.do_y_scale:
            y_pred = self._rev_y_scale(self._predict_inner(x_scaled))
        else:
            y_pred = self._predict_inner(x_scaled)

        return y_pred

    def _predict_inner(self, x_scaled):
        res = []
        # 出力状態計算 & 観測
        for x in x_scaled:
            state = self.circuit.run(x)
            # モデルの出力
            r = [
                self.observables[i].get_expectation_value(state)
                for i in range(self.n_outputs)
            ]  # 出力多次元ver
            res.append(r)
        return np.array(res)

    def cost_func(self, theta, x_scaled, y_scaled):
        if self.cost == "mse":
            self.circuit.update_parameters(theta)
            y_pred = self._predict_inner(x_scaled)

            cost = mean_squared_error(y_pred, y_scaled)
            return cost
        else:
            raise NotImplementedError(
                f"Cost function {self.cost} is not implemented yet."
            )

    def _get_y_scale_param(self, y):
        # 複数入力がある場合に対応したい
        minimum = np.min(y, axis=0)
        maximum = np.max(y, axis=0)
        sa = (maximum - minimum) / 2
        minimum -= sa / self.y_norm_range - 1
        maximum += sa / self.y_norm_range - 1
        sa /= self.y_norm_range
        return [minimum, maximum, sa]

    def _do_y_scale(self, y):
        # yを[-1,1]の範囲に収める
        # print([((ya - self.scale_y_param[0]) / self.scale_y_param[2]) - 1 for ya in y])
        return [((ya - self.scale_y_param[0]) / self.scale_y_param[2]) - 1 for ya in y]

    def _rev_y_scale(self, y_inr):
        # y_inrに含まれる数を、　self.scale_paramを用いて復元する
        return [
            (((ya + 1) * self.scale_y_param[2]) + self.scale_y_param[0]) for ya in y_inr
        ]

    def _cost_func_grad(self, theta, x_scaled, y_scaled):
        self.circuit.update_parameters(theta)

        mto = self._predict_inner(x_scaled).copy()

        grad = np.zeros(len(theta))
        # gradA = np.zeros((len(x_scaled),len(theta)))
        # gradB = np.zeros((len(x_scaled),len(theta)))

        for h in range(len(x_scaled)):
            backobs = Observable(self.n_qubit)
            if self.n_outputs >= 2:
                for i in range(self.n_outputs):
                    backobs.add_operator(
                        (-y_scaled[h][i] + mto[h][i]) / self.n_outputs, "Z {i}"
                    )
            else:
                backobs.add_operator((-y_scaled[h] + mto[h][0]) / self.n_outputs, "Z 0")
            grad += self.circuit.backprop(x_scaled[h], backobs)

        self.circuit.update_parameters(theta)
        grad /= len(x_scaled)
        return grad
