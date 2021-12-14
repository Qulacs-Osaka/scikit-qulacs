from __future__ import annotations

from typing import List, Optional

import numpy as np
from qulacs import Observable
from scipy.optimize import minimize

from skqulacs.circuit import LearningCircuit
from skqulacs.qnn.qnnbase import QNN
from skqulacs.typing import Literal


def _get_x_scale_param(x):
    minimum = np.min(x, axis=0)
    maximum = np.max(x, axis=0)
    sa = (maximum - minimum) / 2
    return [minimum, maximum, sa]


def _min_max_scaling(x: List[List[float]], scale_x_param):
    """[-1, 1]の範囲に規格化"""
    # print([((xa - scale_x_param[0]) / scale_x_param[2]) - 1 for xa in x])
    return [((xa - scale_x_param[0]) / scale_x_param[2]) - 1 for xa in x]


class QNNClassifier(QNN):
    """quantum circuit learningを用いて分類問題を解く"""

    def __init__(
        self,
        n_qubit: int,
        circuit: LearningCircuit,
        num_class: int,
        solver: Literal["BFGS", "Nelder-Mead", "Adam"] = "BFGS",
        cost: Literal["log_loss"] = "log_loss",
        do_x_scale: bool = True,
        y_exp_bai=5.0,
    ) -> None:
        """
        :param nqubit: qubitの数。必要とする出力の次元数よりも多い必要がある
        :param circuit: 回路そのもの
        :param num_class: 分類の数（=測定するqubitの数）
        :param solver: 何を使うか　Nelderは非推奨
        :param cost: コスト関数　log_lossしかない。
        :param do_x_scale xをscaleしますか?
        :param y_exp_bai 内部出力のyが0と1では、　e^y_exp_bai 倍の確率の倍率がある。
        """
        self.n_qubit = n_qubit
        self.circuit = circuit
        self.num_class = num_class  # 分類の数（=測定するqubitの数）
        self.solver = solver
        self.cost = cost
        self.do_x_scale = do_x_scale
        self.y_exp_bai = y_exp_bai
        self.scale_x_param = []
        self.scale_y_param = []  # yのスケーリングのパラメータ

        self.observables = [Observable(n_qubit) for _ in range(n_qubit)]
        for i in range(n_qubit):
            self.observables[i].add_operator(1.0, f"Z {i}")

    def fit(self, x_train, y_train, maxiter: Optional[int] = None):
        """
        :param x_list: fitしたいデータのxのリスト
        :param y_list: fitしたいデータのyのリスト
        :param maxiter: scipy.optimize.minimizeのイテレーション回数
        :return: 学習後のロス関数の値
        :return: 学習後のパラメータthetaの値
        """
        self.scale_x_param = _get_x_scale_param(x_train)
        self.scale_y_param = self.get_y_scale_param(y_train)

        # x_trainからscaleのparamを取得
        # classはyにone-hot表現をする
        # x_scaled = _min_max_scaling(x_train, self.scale_x_param)
        # y_scaled = self.do_y_scale(y_train)
        if self.do_x_scale:
            x_scaled = _min_max_scaling(x_train, self.scale_x_param)
        else:
            x_scaled = x_train

        y_scaled = self.do_y_scale(y_train)

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
            maxiter *= len(x_train)
            for iter in range(0, maxiter, 5):
                grad = self._cost_func_grad(
                    theta_now,
                    x_scaled[iter % len(x_train) : iter % len(x_train) + 5],
                    y_scaled[iter % len(y_train) : iter % len(y_train) + 5],
                )
                moment = moment * pr_Bi + (1 - pr_Bi) * grad
                vel = vel * pr_Bt + (1 - pr_Bt) * np.dot(grad, grad)
                Bix = Bix * pr_Bi + (1 - pr_Bi)
                Btx = Btx * pr_Bt + (1 - pr_Bt)
                theta_now -= pr_A / (((vel / Btx) ** 0.5) + pr_ips) * (moment / Bix)
                # if iter % len(x_train) < 5:
                # self.cost_func(theta_now, x_train, y_train)

            loss = self.cost_func(theta_now, x_train, y_train)
            theta_opt = theta_now
        else:
            raise NotImplementedError
        return loss, theta_opt

    def predict(self, x_test: List[List[float]]):
        """Predict outcome for each input data in `x_test`.

        Arguments:
            x_test: Input data whose shape is (n_samples, n_features).

        Returns:
            y_pred: Predicted outcome.
        """
        # x_test = array-like of of shape (n_samples, n_features)
        if self.do_x_scale:
            x_scaled = _min_max_scaling(x_test, self.scale_x_param)
        else:
            x_scaled = x_test

        y_pred = self.rev_y_scale(self._predict_inner(x_scaled))
        return y_pred

    def _predict_inner(self, x_list):
        # 入力xに関して、量子回路を通した生のデータを表示
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

    def cost_func(self, theta, x_scaled, y_scaled):
        # 生のデータを入れる
        if self.cost == "log_loss":
            # cross-entropy loss (default)
            self.circuit.update_parameters(theta)
            y_pred = self._predict_inner(x_scaled)
            # predについて、softmaxをする
            ypf = []
            for i in range(len(y_pred)):
                for j in range(len(self.scale_y_param[0])):
                    hid = self.scale_y_param[1][j]
                    wa = 0
                    for k in range(self.scale_y_param[0][j]):
                        wa += np.exp(self.y_exp_bai * y_pred[i][hid + k])
                    for k in range(self.scale_y_param[0][j]):
                        ypf.append(np.exp(self.y_exp_bai * y_pred[i][hid + k]) / wa)
            ysf = y_scaled.ravel()
            cost = 0
            for i in range(len(ysf)):
                if ysf[i] == 1:
                    cost -= np.log(ypf[i])
                else:
                    cost -= np.log(1 - ypf[i])
            cost /= len(ysf)
            return cost
        else:
            raise NotImplementedError(
                f"Cost function {self.cost} is not implemented yet."
            )

    def get_y_scale_param(self, y):
        # 複数入力がある場合に対応したい
        # yの最大値をもつ
        syurui = np.max(y, axis=0)
        syurui = syurui.astype(int)
        syurui = syurui + 1
        if not isinstance(syurui, np.ndarray):
            eee = syurui
            syurui = np.zeros(1, dtype=int)
            syurui[0] = eee
        rui = np.concatenate((np.zeros(1, dtype=int), syurui.cumsum()))
        return [syurui, rui]

    def do_y_scale(self, y):
        # yをone-hot表現にする 複数入力への対応もする
        clsnum = int(self.scale_y_param[1][-1])
        res = np.zeros((len(y), clsnum), dtype=int)
        for i in range(len(y)):
            if y.ndim == 1:
                res[i][y[i]] = 1
            else:
                for j in range(len(y[i])):
                    res[i][y[i][j] + self.scale_y_param[1][j]] = 1
        return res

    def rev_y_scale(self, y_inr):
        # argmaxをとる
        # one-hot表現の受け取りをしたら、それを与えられた番号にして返す
        res = np.zeros((len(y_inr), len(self.scale_y_param[0])), dtype=int)
        for i in range(len(y_inr)):
            for j in range(len(self.scale_y_param[0])):
                hid = self.scale_y_param[1][j]
                sai = -9999
                arg = 0
                for k in range(self.scale_y_param[0][j]):
                    if sai < y_inr[i][hid + k]:
                        sai = y_inr[i][hid + k]
                        arg = k
                res[i][j] = arg
        return res

    def _cost_func_grad(self, theta, x_scaled, y_scaled):
        self.circuit.update_parameters(theta)

        mto = self._predict_inner(x_scaled).copy()
        grad = np.zeros(len(theta))
        for h in range(len(x_scaled)):
            for j in range(len(self.scale_y_param[0])):
                hid = self.scale_y_param[1][j]
                wa = 0
                for k in range(self.scale_y_param[0][j]):
                    wa += np.exp(self.y_exp_bai * mto[h][hid + k])
                for k in range(self.scale_y_param[0][j]):
                    mto[h][hid + k] = np.exp(self.y_exp_bai * mto[h][hid + k]) / wa
            backobs = Observable(self.n_qubit)

            for i in range(len(y_scaled[0])):
                if y_scaled[h][i] == 0:
                    backobs.add_operator(1.0 / (1.0 - mto[h][i]), f"Z {i}")
                else:
                    backobs.add_operator(-1.0 / (mto[h][i]), f"Z {i}")
            grad += self.circuit.backprop(x_scaled[h], backobs)

        self.circuit.update_parameters(theta)
        grad /= len(x_scaled)
        return grad
