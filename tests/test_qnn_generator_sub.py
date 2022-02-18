import pandas as pd
import pytest
from sklearn import datasets
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from skqulacs.circuit.pre_defined import create_qcl_ansatz
from skqulacs.qnn import QNNGeneretor
from skqulacs.circuit import create_farhi_neven_ansatz
from skqulacs.circuit import LearningCircuit
def test_generator_func_same():
    #generatorのcost_funcがあってるか検証します
    #2qubit gate 
    circuit = LearningCircuit(3)
    circuit.add_H_gate(0)
    circuit.add_Z_gate(1)
    circuit.add_H_gate(2)
    #circuitの確率は、[0.25,0.25,0,0,0.25,0.25,0,0] のはず
    #それを変化させて[0.5,0.5,0,0]
    #入力するデータは[0.4,0.3,0.2,0.1]
    #つまり、スコアは0.01+0.04+0.04+0.01 = 0.1の誤差のはず
    qcl = QNNGeneretor(circuit, "same", 0,2)

    pre_per=qcl.predict()
    print(pre_per)
    assert abs(pre_per[0]-0.5)< 0.001
    assert abs(pre_per[1]-0.5)< 0.001
    assert abs(pre_per[2]-0.0)< 0.001
    assert abs(pre_per[3]-0.0)< 0.001
    diff_score=qcl.cost_func([],np.array([0.4,0.3,0.2,0.1]))

    assert abs(diff_score-0.1) < 0.001

def test_generator_func_gauss():
    circuit = LearningCircuit(2)
    circuit.add_X_gate(0)
    #circuitの確率は、[0,1,0,0] カーネルを考えると、スコアは1.264241　になるはず
    qcl = QNNGeneretor(circuit, "gauss", 0.5,2)
    diff_score=qcl.cost_func([],np.array([0,0,1,0]))
    assert abs(diff_score-1.264241) < 0.001

def test_generator_func_hamming():
    circuit = LearningCircuit(2)
    #circuitの確率は、[1,0,0,0] カーネルを考えると、スコアは1.264241　になるはず
    qcl = QNNGeneretor(circuit, "exp_hamming", 0.5,2)
    diff_score=qcl.cost_func([],np.array([0,0,1,0]))
    assert abs(diff_score-1.264241) < 0.001

def test_generator_func_gauss2():
    circuit = LearningCircuit(2)
    
    #circuitの確率は、[1,0,0,0] カーネルを考えると、スコアは1.9633687　になるはず
    qcl = QNNGeneretor(circuit, "gauss", 0.5,2)
    diff_score=qcl.cost_func([],np.array([0,0,1,0]))
    assert abs(diff_score-1.9633687) < 0.001

def test_generator_func_hamming2():
    circuit = LearningCircuit(2)
    circuit.add_X_gate(0)
    #circuitの確率は、[0,1,0,0] カーネルを考えると、スコアは1.7293294335267746　になるはず
    qcl = QNNGeneretor(circuit, "exp_hamming", 0.5,2)
    diff_score=qcl.cost_func([],np.array([0,0,1,0]))
    assert abs(diff_score-1.7293294335267746) < 0.001

def test_generator_grad_true():

    n_qubit = 2
    depth = 3
    circuit = create_farhi_neven_ansatz(n_qubit, depth)
    qnn = QNNGeneretor(circuit,"same",0,2)

    gencost=qnn.cost_func([1,1,1,1,1,1,1,1,1,1,1,1],[0.1,0.2,0.3,0.4])
    atocost=qnn.cost_func([1.01,1,1,1,1,1,1,1,1,1,1,1],[0.1,0.2,0.3,0.4])
    gradcost=qnn._cost_func_grad([1,1,1,1,1,1,1,1,1,1,1,1],[0.1,0.2,0.3,0.4])

    print(gencost)
    print(atocost)
    print(gradcost)
    assert abs(gencost+gradcost[0]*0.01*2-atocost)<0.00001
    assert abs(gencost+gradcost[0]*0.01*1-atocost)>0.00001
    assert abs(gencost+gradcost[0]*0.01*3-atocost)>0.00001

