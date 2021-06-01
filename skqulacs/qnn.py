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
def make_fullgate(list_SiteAndOperator, nqubit):
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
			list_SingleGates.append( list_SiteAndOperator[cnt][1] )
			cnt += 1
		else:  # 何もないsiteはidentity
			list_SingleGates.append(I_mat)

	return reduce(np.kron, list_SingleGates)


def create_time_evol_gate(nqubit, time_step=0.77):
	""" ランダム磁場・ランダム結合イジングハミルトニアンをつくって時間発展演算子をつくる
	:param time_step: ランダムハミルトニアンによる時間発展の経過時間
	:return  qulacsのゲートオブジェクト
	"""
	ham = np.zeros((2**nqubit,2**nqubit), dtype = complex)
	for i in range(nqubit):  # i runs 0 to nqubit-1
		Jx = -1. + 2.*np.random.rand()  # -1~1の乱数
		ham += Jx * make_fullgate( [ [i, X_mat] ], nqubit)
		for j in range(i+1, nqubit):
			J_ij = -1. + 2.*np.random.rand()
			ham += J_ij * make_fullgate ([ [i, Z_mat], [j, Z_mat]], nqubit)

	# 対角化して時間発展演算子をつくる. H*P = P*D <-> H = P*D*P^dagger
	diag, eigen_vecs = np.linalg.eigh(ham)
	time_evol_op = np.dot(np.dot(eigen_vecs, np.diag(np.exp(-1j*time_step*diag))), eigen_vecs.T.conj())  # e^-iHT

	# qulacsのゲートに変換
	time_evol_gate = DenseMatrix([i for i in range(nqubit)], time_evol_op)

	return time_evol_gate


def min_max_scaling(x, axis=None):
	"""[-1, 1]の範囲に規格化"""
	min = x.min(axis=axis, keepdims=True)
	max = x.max(axis=axis, keepdims=True)
	result = (x-min)/(max-min)
	result = 2.*result-1.
	return result


def softmax(x):
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
		ham += Jx * make_fullgate([[i, X_mat]],n_qubit)
		for j in range(i + 1, n_qubit):
			J_ij = -1.0 + 2.0 * np.random.rand()
			ham += J_ij * make_fullgate([[i, Z_mat], [j, Z_mat]],n_qubit)
	return ham
class QNNRegressor:
	r"""Solve regression tasks with Quantum Neural Network.

	Args:
		n_qubit: The number of qubits.
		solver: Kind of optimization solver.
		circuit_arch: Form of quantum circuit.
		observable: String representation of observable operators.
		n_shots: The number of measurements to compute expectation.
		cost: Kind of cost function.
	Attributes:
		observable: Observable.
	"""

	OBSERVABLE_COEF: float = 2.0

	def __init__(
		self,
		n_qubit: int,
		circuit_depth: int,
		time_step: float,
		solver: Literal["bfgs"] = "bfgs",
		circuit_arch: Literal["default"] = "default",
		observable: str = "Z 0",
		n_shot: int = np.inf,
		cost: Literal["mse"] = "mse",
	) -> None:
		self.n_qubit = n_qubit
		self.circuit_depth = circuit_depth
		self.time_step = time_step
		self.solver = solver
		self.circuit_arch = circuit_arch
		self.observable = Observable(n_qubit)
		self.observable.add_operator(self.OBSERVABLE_COEF, observable)
		self.n_shot = n_shot
		self.cost = cost
		self.u_out = self._u_output()
		
		self.obs = Observable(self.n_qubit)


	def fit(self, x, y):
		self.obs.add_operator(2.,'Z 0')
		parameter_count = self.u_out.get_parameter_count()
		theta_init_a = [self.u_out.get_parameter(ind) for ind in range(parameter_count)]
		theta_init=theta_init_a.copy()
		#print(theta_init)
		result = minimize(
			self._cost_func, theta_init, args=(x, y), method="Nelder-Mead",
		)
		#print(result.fun)
		theta_opt = result.x
		
		#print(theta_opt)
		self._update_u_out(theta_opt)
		loss = result.fun
		return loss, theta_opt

	def predict(self, x):
		state = QuantumState(self.n_qubit)
		state.set_zero_state()
		self._u_input(x).update_quantum_state(state)
		#print(state)
		self.u_out.update_quantum_state(state)
		#print(state)
		aaa=self.obs.get_expectation_value(state)
		#print(state)
		#print(aaa)
		return aaa

	#@staticmethod
	def _cost_func(self,theta, x_train, y_train):
		self._update_u_out(theta)
		y_predict = [self.predict(x) for x in x_train]
		if self.cost == "mse":
			return ((y_predict - y_train) ** 2).mean()
		else:
			raise NotImplementedError(
				f"Cost function {self.cost} is not implemented yet."
			)

	def _u_input(self, x):
		u_in = QuantumCircuit(self.n_qubit)
		angle_y = np.arcsin(x)
		angle_z = np.arccos(x ** 2)
		for i in range(self.n_qubit):
			u_in.add_RY_gate(i, angle_y)
			u_in.add_RZ_gate(i, angle_z)
		return u_in

	def _u_output(self):
		time_evol_gate = self._time_evol_gate()
		u_out = ParametricQuantumCircuit(self.n_qubit)
		for _ in range(self.circuit_depth):
			u_out.add_gate(time_evol_gate)
			for i in range(self.n_qubit):
				angle = 2.0 * np.pi * np.random.rand()
				u_out.add_parametric_RX_gate(i, angle)
				angle = 2.0 * np.pi * np.random.rand()
				u_out.add_parametric_RZ_gate(i, angle)
				angle = 2.0 * np.pi * np.random.rand()
				u_out.add_parametric_RX_gate(i, angle)
		return u_out

	def _time_evol_gate(self):
		hamiltonian = make_hamiltonian(self.n_qubit)
		diag, eigen_vecs = np.linalg.eigh(hamiltonian)
		time_evol_op = np.dot(
			np.dot(eigen_vecs, np.diag(np.exp(-1j * self.time_step * diag))),
			eigen_vecs.T.conj(),
		)
		return DenseMatrix([i for i in range(self.n_qubit)], time_evol_op)

	def _update_u_out(self, theta):
		param_count = self.u_out.get_parameter_count()
		for i in range(param_count):
			self.u_out.set_parameter(i, theta[i])

	
class QNNClassification:
    """ quantum circuit learningを用いて分類問題を解く"""
    def __init__(self, nqubit, c_depth, num_class):
        """
        :param nqubit: qubitの数。必要とする出力の次元数よりも多い必要がある
        :param c_depth: circuitの深さ
        :param num_class: 分類の数（=測定するqubitの数）
        """
        self.nqubit = nqubit
        self.c_depth = c_depth

        self.input_state_list = []  # |ψ_in>のリスト
        self.theta = []  # θのリスト

        self.output_gate = None  # U_out

        self.num_class = num_class  # 分類の数（=測定するqubitの数）

        # オブザーバブルの準備
        obs = [Observable(nqubit) for _ in range(num_class)]
        for i in range(len(obs)):
            obs[i].add_operator(1., f'Z {i}')  # Z0, Z1, Z3をオブザーバブルとして設定
        self.obs = obs

    def create_input_gate(self, x):
        # 単一のxをエンコードするゲートを作成する関数
        # xは入力特徴量(2次元)
        # xの要素は[-1, 1]の範囲内
        u = QuantumCircuit(self.nqubit)
                
        angle_y = np.arcsin(x)
        angle_z = np.arccos(x**2)

        for i in range(self.nqubit):
            if i % 2 == 0:
                u.add_RY_gate(i, angle_y[0])
                u.add_RZ_gate(i, angle_z[0])
            else:
                u.add_RY_gate(i, angle_y[1])
                u.add_RZ_gate(i, angle_z[1])
        
        return u

    def set_input_state(self, x_list):
        """入力状態のリストを作成"""
        x_list_normalized = min_max_scaling(x_list)  # xを[-1, 1]の範囲にスケール
        
        st_list = []
        
        for x in x_list_normalized:
            st = QuantumState(self.nqubit)
            input_gate = self.create_input_gate(x)
            input_gate.update_quantum_state(st)
            st_list.append(st.copy())
        self.input_state_list = st_list

    def create_initial_output_gate(self):
        """output用ゲートU_outの組み立て&パラメータ初期値の設定"""
        u_out = ParametricQuantumCircuit(self.nqubit)
        time_evol_gate = create_time_evol_gate(self.nqubit)
        theta = 2.0 * np.pi * np.random.rand(self.c_depth, self.nqubit, 3)
        self.theta = theta.flatten()
        for d in range(self.c_depth):
            u_out.add_gate(time_evol_gate)
            for i in range(self.nqubit):
                u_out.add_parametric_RX_gate(i, theta[d, i, 0])
                u_out.add_parametric_RZ_gate(i, theta[d, i, 1])
                u_out.add_parametric_RX_gate(i, theta[d, i, 2])
        self.output_gate = u_out
    
    def update_output_gate(self, theta):
        """U_outをパラメータθで更新"""
        self.theta = theta
        parameter_count = len(self.theta)
        for i in range(parameter_count):
            self.output_gate.set_parameter(i, self.theta[i])

    def get_output_gate_parameter(self):
        """U_outのパラメータθを取得"""
        parameter_count = self.output_gate.get_parameter_count()
        theta = [self.output_gate.get_parameter(ind) for ind in range(parameter_count)]
        return np.array(theta)

    def pred(self, theta):
        """x_listに対して、モデルの出力を計算"""

        # 入力状態準備
        # st_list = self.input_state_list
        st_list = [st.copy() for st in self.input_state_list]  # ここで各要素ごとにcopy()しないとディープコピーにならない
        # U_outの更新
        self.update_output_gate(theta)

        res = []
        # 出力状態計算 & 観測
        for st in st_list:
            # U_outで状態を更新
            self.output_gate.update_quantum_state(st)
            # モデルの出力
            r = [o.get_expectation_value(st) for o in self.obs]  # 出力多次元ver
            r = softmax(r)
            res.append(r.tolist())
        return np.array(res)

    def cost_func(self, theta):
        """コスト関数を計算するクラス
        :param theta: 回転ゲートの角度thetaのリスト
        """

        y_pred = self.pred(theta)

        # cross-entropy loss
        loss = log_loss(self.y_list, y_pred)
        
        return loss

    # for BFGS
    def B_grad(self, theta):
        # dB/dθのリストを返す
        theta_plus = [theta.copy() + np.eye(len(theta))[i] * np.pi / 2. for i in range(len(theta))]
        theta_minus = [theta.copy() - np.eye(len(theta))[i] * np.pi / 2. for i in range(len(theta))]

        grad = [(self.pred(theta_plus[i]) - self.pred(theta_minus[i])) / 2. for i in range(len(theta))]

        return np.array(grad)

    # for BFGS
    def cost_func_grad(self, theta):
        y_minus_t = self.pred(theta) - self.y_list
        B_gr_list = self.B_grad(theta)
        grad = [np.sum(y_minus_t * B_gr) for B_gr in B_gr_list]
        return np.array(grad)

    def fit(self, x_list, y_list, maxiter=200):
        """
        :param x_list: fitしたいデータのxのリスト
        :param y_list: fitしたいデータのyのリスト
        :param maxiter: scipy.optimize.minimizeのイテレーション回数
        :return: 学習後のロス関数の値
        :return: 学習後のパラメータthetaの値

        maxiterが10の倍数でないとき、Iteration countのデバッグが10/10しか表示されない 
        """

        # 初期状態生成
        self.set_input_state(x_list)

        # 乱数でU_outを作成
        self.create_initial_output_gate()
        theta_init = self.theta

        # 正解ラベル
        self.y_list = y_list

        # for callbacks
        self.n_iter = 0
        self.maxiter = maxiter
        
        print("Initial parameter:")
        print(self.theta)
        print()
        print(f"Initial value of cost function:  {self.cost_func(self.theta):.4f}")
        print()
        print('============================================================')
        print("Iteration count...")
        result = minimize(self.cost_func,
                          self.theta,
                          # method='Nelder-Mead',
                          method='BFGS',
                          jac=self.cost_func_grad,
                          options={"maxiter":maxiter},
                          callback=self.callbackF)
        theta_opt = self.theta
        print('============================================================')
        print()
        print("Optimized parameter:")
        print(self.theta)
        print()
        print(f"Final value of cost function:  {self.cost_func(self.theta):.4f}")
        print()
        return result, theta_init, theta_opt

    def callbackF(self, theta):
            self.n_iter = self.n_iter + 1
            if 10 * self.n_iter % self.maxiter == 0:
                print(f"Iteration: {self.n_iter} / {self.maxiter},   Value of cost_func: {self.cost_func(theta):.4f}")
