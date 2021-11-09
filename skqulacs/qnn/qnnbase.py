from abc import ABC, abstractmethod
from functools import reduce
from typing import List, Optional, Tuple
from qulacs.gate import X, Z, DenseMatrix
from numpy.random import Generator, default_rng
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


def _create_time_evol_gate(
    n_qubit, time_step=0.77, rng: Generator = None, seed: int = 0
):
    """ランダム磁場・ランダム結合イジングハミルトニアンをつくって時間発展演算子をつくる
    :param time_step: ランダムハミルトニアンによる時間発展の経過時間
    :return  qulacsのゲートオブジェクト
    """
    if rng is None:
        rng = default_rng(seed)

    ham = make_hamiltonian(n_qubit, rng)
    # 対角化して時間発展演算子をつくる. H*P = P*D <-> H = P*D*P^dagger
    diag, eigen_vecs = np.linalg.eigh(ham)
    time_evol_op = np.dot(
        np.dot(eigen_vecs, np.diag(np.exp(-1j * time_step * diag))), eigen_vecs.T.conj()
    )  # e^-iHT

    # qulacsのゲートに変換
    time_evol_gate = DenseMatrix([i for i in range(n_qubit)], time_evol_op)

    return time_evol_gate


def _get_x_scale_param(x):
    minimum = np.min(x, axis=0)
    maximum = np.max(x, axis=0)
    sa = (maximum - minimum) / 2

    # print(minimum)
    # print(maximum)
    # print(sa)
    # minimum-=sa*0.4
    # maximum+=sa*0.4
    # sa*=1.4
    #print(minimum)
    #print(maximum)
    #print(sa)
    return [minimum, maximum, sa]


def _min_max_scaling(x: List[List[float]], scale_x_param):
    """[-1, 1]の範囲に規格化"""
    # print([((xa - scale_x_param[0]) / scale_x_param[2]) - 1 for xa in x])
    return [((xa - scale_x_param[0]) / scale_x_param[2]) - 1 for xa in x]


def _softmax(x):
    """softmax function
    :param x: ndarray
    """
    y = np.exp(x) / np.sum(np.exp(x))
    return y


def make_hamiltonian(n_qubit, rng: Generator = None, seed: int = 0):
    if rng is None:
        rng = default_rng(seed)

    ham = np.zeros((2 ** n_qubit, 2 ** n_qubit), dtype=complex)
    for i in range(n_qubit):
        Jx = rng.uniform(-1.0, 1.0)
        ham += Jx * _make_fullgate([[i, X_mat]], n_qubit)
        for j in range(i + 1, n_qubit):
            J_ij = rng.uniform(-1.0, 1.0)
            ham += J_ij * _make_fullgate([[i, Z_mat], [j, Z_mat]], n_qubit)
    return ham


class QNN(ABC):
    @abstractmethod
    def fit(
        self, x_train: List[float], y_train: List[float], maxiter: Optional[int]
    ) -> Tuple[float, np.ndarray]:
        """Fit the model to given train data.

        Args:
            x_train: Train data of independent variable.
            y_train: Train data of dependent variable.
            maxiter: Maximum number of iterations for a cost minimization solver.

        Returns:
            loss: Loss of minimized cost function.
            theta_opt: Parameter of optimized model.
        """
        pass

    @abstractmethod
    def predict(self, theta: List[float], x_list: List[float]) -> List[float]:
        """Predict outcome for given data.

        Args:
            theta: Parameter of model. For most cases, give `theta_opt` from `QNN.fit`.
            x_list: Input data to predict outcome.

        Returns:
            y_pred: List of predicted data. `y_pred[i]` corresponds to `x_list[i]`.
        """
        pass
