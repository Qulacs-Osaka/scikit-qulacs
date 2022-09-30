from __future__ import annotations

from math import exp, sqrt
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from qulacs import Observable, QuantumState
from qulacs.gate import DenseMatrix
from typing_extensions import Literal

from skqulacs.circuit import LearningCircuit
from skqulacs.qnn.solver import Solver


class QNNGeneretor:
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
        solver: Solver,
        karnel_type: Literal["gauss", "exp_hamming", "same"],
        gauss_sigma: float,
        fitting_qubit: int,
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
        self.fitting_qubit = fitting_qubit
        self.karnel_type = karnel_type
        self.gauss_sigma = gauss_sigma
        self.solver = solver

        self.observables = [Observable(self.n_qubit) for _ in range(self.n_qubit)]
        for i in range(self.n_qubit):
            self.observables[i].add_operator(1.0, f"Z {i}")

    def fit(
        self, train_data, maxiter: Optional[int] = None
    ) -> Tuple[float, List[float]]:
        """
        :param train_scaled: trainの確率分布を入力

        :param maxiter: scipy.optimize.minimizeのイテレーション回数
        :return: 学習後のロス関数の値
        :return: 学習後のパラメータthetaの値
        """
        train_scaled = np.zeros(2**self.fitting_qubit)
        for i in train_data:
            train_scaled[i] += 1 / len(train_data)
        return self.fit_direct_distribution(train_scaled, maxiter)

    def fit_direct_distribution(
        self, train_scaled, maxiter: Optional[int] = None
    ) -> Tuple[float, List[float]]:
        theta_init = self.circuit.get_parameters()
        return self.solver.run(
            self.cost_func,
            self._cost_func_grad,
            theta_init,
            train_scaled,
            [],
            maxiter,
        )

    def predict(self) -> NDArray[np.float_]:
        """
        予想される確率分布を、np.ndarray[float]の形で返す。

        Returns:
            data_per: Predicted distribution.
        """

        y_pred_in = self._predict_inner().get_vector()
        y_pred_conj = y_pred_in.conjugate()

        data_per = y_pred_in * y_pred_conj  # 2乗の和

        if self.n_qubit != self.fitting_qubit:  # いくつかのビットを捨てる
            data_per = data_per.reshape(
                (2 ** (self.n_qubit - self.fitting_qubit), 2**self.fitting_qubit)
            )
            data_per = data_per.sum(axis=0)

        return data_per

    def _predict_inner(self) -> QuantumState:
        # 入力xに関して、量子回路を通した生のデータを表示
        # 出力状態計算 & 観測
        state = self.circuit.run([0])
        return state

    def _predict_and_inner(self) -> Tuple[NDArray[np.float_], QuantumState]:
        # Necessary because `cost_func_grad` needs a state created in prediction.
        state = self._predict_inner()
        y_pred_in = state.get_vector()
        y_pred_conj = y_pred_in.conjugate()

        data_per = y_pred_in * y_pred_conj  # 2乗の和

        if self.n_qubit != self.fitting_qubit:  # いくつかのビットを捨てる
            data_per = data_per.reshape(
                (2 ** (self.n_qubit - self.fitting_qubit), 2**self.fitting_qubit)
            )
            data_per = data_per.sum(axis=0)

        return (data_per, state)

    def conving(self, data_diff):
        # data_diffは、現在の分布ー正しい分布
        # (data_diff) (カーネル行列) (data_diffの行ベクトル)を計算すると、cost_funcになる。
        # ここでは、(data_diff) (カーネル行列)  のベクトルを求める。
        # つまり、確率差ベクトルにカーネル行列を掛ける。
        if self.karnel_type == "gauss":
            # 高速化として、|x-y|=4√σ を超える場合(つまりkがexp(-8)=0.000335以下)は打ち切る。
            beta = -0.5 / self.gauss_sigma

            width = int(4 * sqrt(self.gauss_sigma))
            conv_len = width * 2 + 1
            conv_target = np.zeros(conv_len)
            for i in range(conv_len):
                conv_target[i] = exp((i - width) * (i - width) * beta)

            if conv_len <= 2**self.fitting_qubit:
                conv_diff = np.convolve(data_diff, conv_target, mode="same")
            else:
                # convの行列のほうが長いので、sameがバグった。
                # だから、わざわざfullのを取って、スライスしている。
                conv_diff = np.convolve(data_diff, conv_target)[width:-width]

            return conv_diff

        elif self.karnel_type == "exp_hamming":

            beta = -0.5 / self.gauss_sigma
            swap_pena = exp(beta)
            # ハミング距離の畳み込み演算をします。
            # これはバタフライ演算でできます。
            # バタフライ演算を行うのにqulacsを使います。
            # 注意！ユニタリ的な量子演算ではありません！
            # この演算をすることで高速にできる。

            diff_state = QuantumState(self.fitting_qubit)
            diff_state.load(data_diff)
            for i in range(self.fitting_qubit):
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

    def cost_func(self, theta, train_scaled, _unuse=[]):
        self.circuit.update_parameters(theta)
        # y-xを求める
        data_diff = self.predict() - train_scaled
        conv_diff = self.conving(data_diff)
        return np.dot(data_diff, conv_diff)

    def _cost_func_grad(self, theta, train_scaled, _unuse=[]):
        self.circuit.update_parameters(theta)
        # y-xを求める
        (pre, prein) = self._predict_and_inner()
        data_diff = pre - train_scaled
        conv_diff = self.conving(data_diff)

        convconv_diff = np.tile(
            conv_diff, 2 ** (self.n_qubit - self.fitting_qubit)
        )  # 得られた確率ベクトルの添え字の大きい桁を無視する。
        # 例: [0.1,0.3,-0.2,0.1  ,  0.1,-0.4,0.2,-0.2] -> [0.2,-0.1,0,-0.1]
        state_vec = prein.get_vector()
        ret = QuantumState(self.n_qubit)
        ret.load(convconv_diff * state_vec * 4)
        # 各要素ごとに積を取り、4を掛けている。4なのは、2乗だから2をかけるのと、実際はカーネルの左と右両方にベクトルあるから2を掛ける。
        return np.array(self.circuit.backprop_inner_product([0], ret))
