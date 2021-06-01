from __future__ import annotations
from functools import reduce
from qulacs import QuantumState, QuantumCircuit, ParametricQuantumCircuit, Observable
from qulacs.gate import X, Z, DenseMatrix
from scipy.optimize import minimize
from typing import Literal
import numpy as np
from qulacs import Observable

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
		print(theta_init)
		result = minimize(
			self._cost_func, theta_init, args=(x, y), method="Nelder-Mead",
		)
		print(result.fun)
		theta_opt = result.x
		
		print(theta_opt)
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
		hamiltonian = self._make_hamiltonian()
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

	def _make_fullgate(self, site_and_operators):
		site = [site_and_operator[0] for site_and_operator in site_and_operators]
		single_gates = []
		count = 0
		I_mat = np.eye(2, dtype=complex)
		for i in range(self.n_qubit):
			if i in site:
				single_gates.append(site_and_operators[count][1])
				count += 1
			else:
				single_gates.append(I_mat)
		return reduce(np.kron, single_gates)

	def _make_hamiltonian(self):
		ham = np.zeros((2 ** self.n_qubit, 2 ** self.n_qubit), dtype=complex)
		X_mat = X(0).get_matrix()
		Z_mat = Z(0).get_matrix()
		for i in range(self.n_qubit):
			Jx = -1.0 + 2.0 * np.random.rand()
			ham += Jx * self._make_fullgate([[i, X_mat]])
			for j in range(i + 1, self.n_qubit):
				J_ij = -1.0 + 2.0 * np.random.rand()
				ham += J_ij * self._make_fullgate([[i, Z_mat], [j, Z_mat]])
		return ham