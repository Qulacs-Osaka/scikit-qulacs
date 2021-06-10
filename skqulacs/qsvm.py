import numpy as np
from qulacs import QuantumState,QuantumCircuit, Observable, PauliOperator
from qulacs.gate import X,Z,RX,RY,RZ,CNOT,merge,DenseMatrix,add
from qulacs.state import inner_product
import matplotlib.pyplot as plt
from sklearn import svm
class QSVR:
	def __init__(
		self,
		tlotstep:int =4
	)->None:
		self.regr=SVM.SVR(kernel="precomputed")
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
			data_state = QuantumState(nqubits)
			data_state.set_zero_state()
			for tlotkai in range(tlotstep):
				for a in range(n_qubit):
					Z(a,x[i][a]/tlotstep).update_quantum_state(data_state)
					#aとa+1のゲートの交互作用
					b=(a+1)%n_qubit
					CNOT(a,b).update_quantum_state(data_state)
					Z(b,(np.pi-x[i][a])*(np.pi-x[i][b])/tlotstep).update_quantum_state(data_state)
					CNOT(a,b).update_quantum_state(data_state)
			#000の行のベクトルを取る
			data_states.append(data_state)
		for i in range(len(x)):
			for j in range(len(x)):
				kar[i][j]=abs(inner_product(data_states[i],data_states[j]))**2
		regr.fit(kar,y)

	
	def predict(self,x):
		regr.predict(x)
	
		
	