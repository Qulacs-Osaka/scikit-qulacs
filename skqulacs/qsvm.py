import numpy as np
from qulacs import QuantumState,QuantumCircuit, Observable, PauliOperator
from qulacs.gate import H,X,Z,RX,RY,RZ,CNOT,merge,DenseMatrix,add
from qulacs.state import inner_product
import matplotlib.pyplot as plt
from sklearn import svm


def get_qvec(x,n_qubit,tlotstep):
	#xはデータ
	#n_qubit,tlotstepはそのままの意味
	data_state = QuantumState(n_qubit)
	data_state.set_zero_state()
	for a in range(n_qubit):
		H(a).update_quantum_state(data_state)
	for tlotkai in range(tlotstep):
		for a in range(n_qubit):
			RZ(a,x[a]/tlotstep).update_quantum_state(data_state)
			#aとa+1のゲートの交互作用
			b=(a+1)%n_qubit
			CNOT(a,b).update_quantum_state(data_state)
			RZ(b,(np.pi-x[a])*(np.pi-x[b])/tlotstep).update_quantum_state(data_state)
			CNOT(a,b).update_quantum_state(data_state)
	for a in range(n_qubit):
		H(a).update_quantum_state(data_state)
	for tlotkai in range(tlotstep):
		for a in range(n_qubit):
			RZ(a,x[a]/tlotstep).update_quantum_state(data_state)
			#aとa+1のゲートの交互作用
			b=(a+1)%n_qubit
			CNOT(a,b).update_quantum_state(data_state)
			RZ(b,(np.pi-x[a])*(np.pi-x[b])/tlotstep).update_quantum_state(data_state)
			CNOT(a,b).update_quantum_state(data_state)
	#000の行のベクトルを取る
	return data_state
class QSVR:
	def __init__(
		self,
		tlotstep:int =4
	)->None:
		self.regr=svm.SVR(kernel="precomputed")
		self.data_states=[]
		self.n_qubit=0
		self.tlotstep=tlotstep
		


	def fit(self, x, y):
		self.n_qubit=len(x[0])
		kar=np.zeros((len(x),len(x))) # サンプル数の二乗の情報量　距離を入れる
		#xとyのカーネルを計算する
		#そのために、UΦxを計算する
		#expを含む計算なので、トロッター法を使って計算する
		#その後、|Φx> = UΦx H  UΦx H |0>^n を行う
		#その後、　x[i]とx[j]]の類似度は、　内積をとって計算する
		for i in range(len(x)):
			self.data_states.append(get_qvec(x[i],self.n_qubit,self.tlotstep))
		for i in range(len(x)):
			for j in range(len(x)):
				kar[i][j]=abs(inner_product(self.data_states[i],self.data_states[j]))**2
		self.regr.fit(kar,y)

	
	def predict(self,xs):
		kar=np.zeros((len(xs),len(self.data_states))) # サンプル数の二乗の情報量　距離を入れる
		for i in range(len(xs)):
			x_qc=get_qvec(xs[i],self.n_qubit,self.tlotstep)
			for j in range(len(self.data_states)):
				kar[i][j]=abs(inner_product(x_qc,self.data_states[j]))**2
		return self.regr.predict(kar)
		


class QSVC:
	def __init__(
		self,
		tlotstep:int =4
	)->None:
		self.regr=svm.SVC(kernel="precomputed")
		self.data_states=[]
		self.n_qubit=0
		self.tlotstep=tlotstep
		


	def fit(self, x, y):
		print(x)
		self.n_qubit=len(x[0])
		kar=np.zeros((len(x),len(x))) # サンプル数の二乗の情報量　距離を入れる
		#xとyのカーネルを計算する
		#そのために、UΦxを計算する
		#expを含む計算なので、トロッター法を使って計算する
		#その後、|Φx> = UΦx H  UΦx H |0>^n を行う
		#その後、　x[i]とx[j]]の類似度は、　内積をとって計算する
		for i in range(len(x)):
			self.data_states.append(get_qvec(x[i],self.n_qubit,self.tlotstep))
			
		for i in range(len(x)):
			print(self.data_states[i])
			for j in range(len(x)):
				kar[i][j]=abs(inner_product(self.data_states[i],self.data_states[j]))**2
				
		print(kar)
		self.regr.fit(kar,y)

	
	def predict(self,xs):
		kar=np.zeros((len(xs),len(self.data_states))) # サンプル数の二乗の情報量　距離を入れる
		for i in range(len(xs)):
			x_qc=get_qvec(xs[i],self.n_qubit,self.tlotstep)
			for j in range(len(self.data_states)):
				kar[i][j]=abs(inner_product(x_qc,self.data_states[j]))**2
		print(kar)
		return self.regr.predict(kar)