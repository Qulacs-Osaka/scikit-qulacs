from __future__ import annotations
from functools import reduce
from skqulacs.qnn.qnnbase import (
    QNN,
    _make_fullgate,
    _create_time_evol_gate,
    _min_max_scaling,
    _softmax,
    make_hamiltonian,
)
from qulacs import QuantumState, QuantumCircuit, ParametricQuantumCircuit, Observable
from scipy.sparse.construct import rand
from qulacs.gate import X, Z, DenseMatrix
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from numpy.random import RandomState
import numpy as np

# 基本ゲート
I_mat = np.eye(2, dtype=complex)
X_mat = X(0).get_matrix()
Z_mat = Z(0).get_matrix()


class QNNClassification(QNN):
    """quantum circuit learningを用いて分類問題を解く"""

    def __init__(self, n_qubit: int, circuit_depth: int, num_class: int, seed: int = 0):
        """
        :param nqubit: qubitの数。必要とする出力の次元数よりも多い必要がある
        :param c_depth: circuitの深さ
        :param num_class: 分類の数（=測定するqubitの数）
        """
        self.n_qubit = n_qubit
        self.circuit_depth = circuit_depth
        self.num_class = num_class  # 分類の数（=測定するqubitの数）
        self.input_state_list = []  # |ψ_in>のリスト
        self.theta_list = []
        self.random_state = RandomState(seed)
        self.output_gate = self._create_initial_output_gate()  # U_out

        # オブザーバブルの準備
        obs = [Observable(n_qubit) for _ in range(num_class)]
        for i in range(len(obs)):
            obs[i].add_operator(1.0, f"Z {i}")  # Z0, Z1, Z3をオブザーバブルとして設定
        self.obs = obs

    def fit(self, x_train, y_train, maxiter: int = 100):
        """
        :param x_list: fitしたいデータのxのリスト
        :param y_list: fitしたいデータのyのリスト
        :param maxiter: scipy.optimize.minimizeのイテレーション回数
        :return: 学習後のロス関数の値
        :return: 学習後のパラメータthetaの値
        """
        # 初期状態生成
        self._set_input_state(x_train)
        # 乱数でU_outを作成
        self._create_initial_output_gate()
        # 正解ラベル
        # one-hot 表現 shape:(150, 3)
        self.y_list = np.eye(3)[y_train]
        theta_init = self.theta_list

        result = minimize(
            self.cost_func,
            theta_init,
            args=(x_train,),
            method="BFGS",
            jac=self._cost_func_grad,
            options={"maxiter": maxiter},
        )
        loss = result.fun
        theta_opt = result.x
        return loss, theta_opt

    def predict(self, theta, x):
        y_pred = self._predict_inner(theta, x)
        y_pred = list(map(np.argmax, y_pred))
        return y_pred

    def _predict_inner(self, theta, x):
        """x_listに対して、モデルの出力を計算"""
        # 入力状態準備
        # st_list = self.input_state_list
        # ここで各要素ごとにcopy()しないとディープコピーにならない
        self._update_output_gate(theta)
        self._set_input_state(x)
        state_list = [state.copy() for state in self.input_state_list]

        res = []
        # 出力状態計算 & 観測
        for state in state_list:
            # U_outで状態を更新
            self.output_gate.update_quantum_state(state)
            # モデルの出力
            r = [obs.get_expectation_value(state) for obs in self.obs]  # 出力多次元ver
            r = _softmax(r)
            res.append(r.tolist())
        return np.array(res)

    def _set_input_state(self, x_list):
        """入力状態のリストを作成"""
        x_list_normalized = _min_max_scaling(x_list)  # xを[-1, 1]の範囲にスケール

        state_list = []
        for x in x_list_normalized:
            input_gate = self._create_input_gate(x)
            state = QuantumState(self.n_qubit)
            input_gate.update_quantum_state(state)
            state_list.append(state)
        self.input_state_list = state_list

    def _create_input_gate(self, x):
        # 単一のxをエンコードするゲートを作成する関数
        # xは入力特徴量(2次元)
        # xの要素は[-1, 1]の範囲内
        u_in = QuantumCircuit(self.n_qubit)
        angle_y = np.arcsin(x)
        angle_z = np.arccos(x ** 2)

        for i in range(self.n_qubit):
            u_in.add_RY_gate(i, angle_y[i % len(angle_y)])
            u_in.add_RZ_gate(i, angle_z[i % len(angle_z)])

        return u_in

    def _create_initial_output_gate(self):
        """output用ゲートU_outの組み立て&パラメータ初期値の設定"""
        u_out = ParametricQuantumCircuit(self.n_qubit)
        time_evol_gate = _create_time_evol_gate(
            self.n_qubit, random_state=self.random_state
        )
        num_parametric_gate = 3
        theta = (
            2.0
            * np.pi
            * self.random_state.rand(
                self.circuit_depth, self.n_qubit, num_parametric_gate
            )
        )
        self.theta_list = theta.flatten()
        for d in range(self.circuit_depth):
            u_out.add_gate(time_evol_gate)
            for i in range(self.n_qubit):
                u_out.add_parametric_RX_gate(i, theta[d, i, 0])
                u_out.add_parametric_RZ_gate(i, theta[d, i, 1])
                u_out.add_parametric_RX_gate(i, theta[d, i, 2])
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

    def cost_func(self, theta, x_train):
        """コスト関数を計算するクラス
        :param theta: 回転ゲートの角度thetaのリスト
        """
        y_pred = self._predict_inner(theta, x_train)
        # cross-entropy loss
        return log_loss(self.y_list, y_pred)

    # for BFGS
    def _cost_func_grad(self, theta, x_train):
        y_minus_t = self._predict_inner(theta, x_train) - self.y_list
        B_grad_list = self._b_grad(theta, x_train)
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
