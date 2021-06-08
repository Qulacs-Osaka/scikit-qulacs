from __future__ import annotations
from functools import reduce
from qulacs import QuantumState, QuantumCircuit, ParametricQuantumCircuit, Observable
from qulacs.gate import X, Z, DenseMatrix
from scipy.optimize import minimize
from typing import Literal
import numpy as np
from qulacs import Observable
from sklearn.metrics import log_loss

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
    time_evol_gate = DenseMatrix([i for i in range(nqubit)], time_evol_op)

    return time_evol_gate


def _min_max_scaling(x, axis=None):
    """[-1, 1]の範囲に規格化"""
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x - min) / (max - min)
    result = 2.0 * result - 1.0
    return result


def _softmax(x):
    """softmax function
    :param x: ndarray
    """
    exp_x = np.exp(x)
    y = exp_x / np.sum(np.exp(x))
    return y


def make_hamiltonian(n_qubit):
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


class QNNClassification:
    """quantum circuit learningを用いて分類問題を解く"""

    def __init__(self, n_qubit: int, circuit_depth: int, num_class: int):
        """
        :param nqubit: qubitの数。必要とする出力の次元数よりも多い必要がある
        :param c_depth: circuitの深さ
        :param num_class: 分類の数（=測定するqubitの数）
        """
        self.n_qubit = n_qubit
        self.circuit_depth = circuit_depth
        self.input_state_list = []  # |ψ_in>のリスト
        self.output_gate = self._create_initial_output_gate()  # U_out
        self.num_class = num_class  # 分類の数（=測定するqubitの数）

        # オブザーバブルの準備
        obs = [Observable(n_qubit) for _ in range(num_class)]
        for i in range(len(obs)):
            obs[i].add_operator(1.0, f"Z {i}")  # Z0, Z1, Z3をオブザーバブルとして設定
        self.obs = obs

    def fit(self, x_train, y_train, maxiter=200):
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
        self.y_list = y_train
        parameter_count = self.output_gate.get_parameter_count()
        theta_init = list(map(self.output_gate.get_parameter, range(parameter_count)))

        result = minimize(
            self.cost_func,
            theta_init,
            args=(x_train,),
            method="BFGS",
            jac=self._cost_func_grad,
            options={"maxiter": maxiter},
        )
        theta_opt = result.x
        loss = result.fun
        return loss, theta_opt

    def predict(self, theta, x):
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
            r = [o.get_expectation_value(state) for o in self.obs]  # 出力多次元ver
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
            # copy() は省略できないか?
            state_list.append(state.copy())
        self.input_state_list = state_list

    def _create_input_gate(self, x):
        # 単一のxをエンコードするゲートを作成する関数
        # xは入力特徴量(2次元)
        # xの要素は[-1, 1]の範囲内
        u_in = QuantumCircuit(self.n_qubit)
        angle_y = np.arcsin(x)
        angle_z = np.arccos(x ** 2)

        for i in range(self.n_qubit):
            if i % 2 == 0:
                u_in.add_RY_gate(i, angle_y[0])
                u_in.add_RZ_gate(i, angle_z[0])
            else:
                u_in.add_RY_gate(i, angle_y[1])
                u_in.add_RZ_gate(i, angle_z[1])

        return u_in

    def _create_initial_output_gate(self):
        """output用ゲートU_outの組み立て&パラメータ初期値の設定"""
        u_out = ParametricQuantumCircuit(self.n_qubit)
        time_evol_gate = _create_time_evol_gate(self.n_qubit)
        theta = 2.0 * np.pi * np.random.rand(self.circuit_depth, self.n_qubit, 3)
        for d in range(self.circuit_depth):
            u_out.add_gate(time_evol_gate)
            for i in range(self.n_qubit):
                u_out.add_parametric_RX_gate(i, theta[d, i, 0])
                u_out.add_parametric_RZ_gate(i, theta[d, i, 1])
                u_out.add_parametric_RX_gate(i, theta[d, i, 2])
        return u_out

    def _update_output_gate(self, theta):
        """U_outをパラメータθで更新"""
        parameter_count = len(theta)
        for i in range(parameter_count):
            self.output_gate.set_parameter(i, theta[i])

    def _get_output_gate_parameter(self):
        """U_outのパラメータθを取得"""
        parameter_count = self.output_gate.get_parameter_count()
        theta = [self.output_gate.get_parameter(ind) for ind in range(parameter_count)]
        return np.array(theta)

    def cost_func(self, theta, x_train):
        """コスト関数を計算するクラス
        :param theta: 回転ゲートの角度thetaのリスト
        """
        y_pred = self.predict(theta, x_train)
        # cross-entropy loss
        return log_loss(self.y_list, y_pred)

    # for BFGS
    def _cost_func_grad(self, theta, x_train):
        y_minus_t = self.predict(theta, x_train) - self.y_list
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
                self.predict(theta_plus[i], x_train)
                - self.predict(theta_minus[i], x_train)
            )
            / 2.0
            for i in range(len(theta))
        ]

        return np.array(grad)
