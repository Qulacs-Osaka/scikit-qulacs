from __future__ import annotations

from math import exp, sqrt
from typing import Literal, Optional

import numpy as np
from qulacs import Observable, QuantumState
from qulacs.gate import DenseMatrix
from scipy.optimize import minimize

from skqulacs.circuit import LearningCircuit
from skqulacs.qnn.qnnbase import QNN


class QNNGeneretor(QNN):
    """
    quantum circuit learningを用いて生成モデルをやる

    入力無し、出力される確率分布をテストデータの確率分布に近くするのが目的

    入力は、データの配列.入力はビット列を2進表記することによる整数しか受け取らない。

    fit_direct_distributionは、直接確率分布を入力する。


    出力は、予想される確率分布を返す。
    """

    def __init__(
        self,
        circuit: LearningCircuit,
        karnel_type: Literal["gauss", "exp_hamming", "same"],
        gauss_sigma: float,
        fitting_qubit: int,
        solver: Literal["Adam", "BFGS"] = "BFGS",
    ) -> None:
        """
        :param circuit: 回路そのもの

        :param fitting_qubit: circuitのqubitとは別で、データのqubit数を示す。
        circuitがデータのビットより大きいと、隠れ層的な役割でうまくいく可能性がある。
        fitting_qubitがcircuitのqubitより大きい場合、出力のビット番号が大きい側は無視される。
        fitting_qubitがcircuitのqubitより小さい場合、多分バグる。


        :param kernel_type 2つの出力がどのくらい似ているかのカーネルっぽいものを指定します。(うまく説明できない)
        gauss_sigmaで詳しく説明します。

        :param gauss_sigma カーネルはガウシアン関数で測られるが、そのシグマ値。
        sameの場合、数字はなんでもいい。

        gaussの場合、k(x,y)= exp((-1/2σ)  (x-y)^2)
        exp_hammingの場合、k(x,y)= exp((-1/2σ)  (xとyのハミング距離))
        sameを指定した場合、k(x,y)= if(x==y):1 else 0

        """
        self.n_qubit = circuit.n_qubit
        self.circuit = circuit
        self.Fqubit = fitting_qubit
        self.karnel_type = karnel_type
        self.gauss_sigma = gauss_sigma
        self.solver = solver

        self.observables = [Observable(self.n_qubit) for _ in range(self.n_qubit)]
        for i in range(self.n_qubit):
            self.observables[i].add_operator(1.0, f"Z {i}")

    def fit(self, train_data, maxiter: Optional[int] = None):
        """
        :param train_scaled: trainの確率分布を入力

        :param maxiter: scipy.optimize.minimizeのイテレーション回数
        :return: 学習後のロス関数の値
        :return: 学習後のパラメータthetaの値
        """
        train_scaled = np.zeros(2 ** self.Fqubit)
        for aaa in train_data:
            train_scaled[aaa] += 1 / len(train_data)
        return self.fit_direct_distribution(train_scaled, maxiter)

    def fit_direct_distribution(self, train_scaled, maxiter: Optional[int] = None):
        theta_init = self.circuit.get_parameters()
        if self.solver == "Adam":
            pr_A = 0.03
            pr_Bi = 0.8
            pr_Bt = 0.995
            pr_ips = 0.0000001
            # ここまでがハイパーパラメータ
            Bix = 0
            Btx = 0

            moment = np.zeros(len(theta_init))
            vel = 0
            theta_now = theta_init
            # print(train_scaled)
            if maxiter is None:
                maxiter = 50
            for iter in range(0, maxiter):
                grad = self._cost_func_grad(theta_now, train_scaled)

                moment = moment * pr_Bi + (1 - pr_Bi) * grad
                vel = vel * pr_Bt + (1 - pr_Bt) * np.dot(grad, grad)
                Bix = Bix * pr_Bi + (1 - pr_Bi)
                Btx = Btx * pr_Bt + (1 - pr_Bt)
                theta_now -= pr_A / (((vel / Btx) ** 0.5) + pr_ips) * (moment / Bix)
                # if iter % len(x_train) < 5:
                # self.cost_func(theta_now, x_train, y_train)

            loss = self.cost_func(theta_now, train_scaled)
            theta_opt = theta_now
        elif self.solver == "BFGS":
            result = minimize(
                self.cost_func,
                theta_init,
                args=train_scaled,
                method=self.solver,
                jac=self._cost_func_grad,
                options={"maxiter": maxiter},
            )
            # print(self._cost_func_grad(result.x, x_scaled, y_scaled))
            loss = result.fun
            theta_opt = result.x
        else:
            raise NotImplementedError
        return loss, theta_opt

    def predict(self):
        """
        予想される確率分布を、np.ndarray[float]の形で返す。

        Returns:
            data_per: Predicted distribution.
        """

        y_pred_in = self._predict_inner().get_vector()
        y_pred_conj = y_pred_in.conjugate()

        data_per = y_pred_in * y_pred_conj  # 2乗の和

        if self.n_qubit != self.Fqubit:  # いくつかのビットを捨てる
            data_per = data_per.reshape(
                (2 ** (self.n_qubit - self.Fqubit), 2 ** self.Fqubit)
            )
            data_per = data_per.sum(axis=0)

        return data_per

    def _predict_inner(self):
        # 入力xに関して、量子回路を通した生のデータを表示
        # 出力状態計算 & 観測
        state = self.circuit.run([0])
        return state

    def conving(self, data_diff):
        # data_diffは、現在の分布ー正しい分布
        # (data_diff) (カーネル行列) (data_diffの行ベクトル)　を計算すると、cost_funcになる。
        # ここでは、(data_diff) (カーネル行列)  のベクトルを求める。
        # 　つまり、確率差ベクトルにカーネル行列を掛ける。
        if self.karnel_type == "gauss":
            # 高速化として、|x-y|=4√σ を超える場合(つまりkがexp(-8)=0.000335以下)は打ち切る。
            beta = -0.5 / self.gauss_sigma

            miru = int(4 * sqrt(self.gauss_sigma))
            conv_aite = np.zeros(miru + miru + 1)
            for i in range(miru + miru + 1):
                conv_aite[i] = exp((i - miru) * (i - miru) * beta)

            if miru + miru + 1 <= 2 ** self.Fqubit:
                conv_diff = np.convolve(data_diff, conv_aite, mode="same")
            else:
                # convの行列のほうが長いので、sameがバグった。
                # だから、わざわざfullのを取って、スライスしている。
                conv_diff = np.convolve(data_diff, conv_aite)[miru:-miru]

            return conv_diff

        elif self.karnel_type == "exp_hamming":

            beta = -0.5 / self.gauss_sigma
            swap_pena = exp(beta)
            # ハミング距離の畳み込み演算をします。
            # これはバタフライ演算でできます。
            # バタフライ演算を行うのにqulacsを使います。
            # 注意！ユニタリ的な量子演算ではありません！
            # この演算をすることで高速にできる。

            diff_state = QuantumState(self.Fqubit)
            diff_state.load(data_diff)
            for i in range(self.Fqubit):
                batafly_gate = DenseMatrix(i, [[1, swap_pena], [swap_pena, 1]])
                batafly_gate.update_quantum_state(diff_state)

            conv_diff = diff_state.get_vector()
            return conv_diff

        elif self.karnel_type == "same":
            return data_diff
        else:
            raise NotImplementedError(
                f"Cost function {self.cost} is not implemented yet."
            )

    def cost_func(self, theta, train_scaled):
        self.circuit.update_parameters(theta)
        # y-xを求める
        data_diff = self.predict() - train_scaled
        conv_diff = self.conving(data_diff)
        return np.dot(data_diff, conv_diff)

    def _cost_func_grad(self, theta, train_scaled):
        self.circuit.update_parameters(theta)
        # y-xを求める
        data_diff = self.predict() - train_scaled
        conv_diff = self.conving(data_diff)

        convconv_diff = np.tile(
            conv_diff, 2 ** (self.n_qubit - self.Fqubit)
        )  # 得られた確率ベクトルの添え字の大きい桁を無視する。
        # 例: [0.1,0.3,-0.2,0.1  ,  0.1,-0.4,0.2,-0.2] -> [0.2,-0.1,0,-0.1]

        state_vec = self._predict_inner().get_vector()
        ret = QuantumState(self.n_qubit)
        ret.load(convconv_diff * state_vec * 4)
        # 各要素ごとに積を取り、4を掛けている。　4なのは、2乗だから2をかけるのと、　実際はカーネルの左と右両方にベクトルあるから2を掛ける。
        return np.array(self.circuit.backprop_inner_product([0], ret))