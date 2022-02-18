from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.random import default_rng
from sklearn.metrics import mean_squared_error
from math import exp
from skqulacs.circuit import create_farhi_neven_ansatz
from skqulacs.qnn import QNNGeneretor
import random
def test_mix_gauss():

    n_qubit = 10
    depth = 10
    time_step = 0.5
    circuit = create_farhi_neven_ansatz(n_qubit, depth)
    qnn = QNNGeneretor(circuit,"gauss",10,10)

    #100000個のデータを作る
    prob_list=np.zeros(1024)
    ua=1024*2/7
    ub=1024*5/7
    v=1024*1/8
    prob_sum=0
    for i in range(1024):
        prob_list[i]=exp(-(ua-i)*(ua-i)/(2*v*v))+exp(-(ub-i)*(ub-i)/(2*v*v))
        prob_sum+=prob_list[i]
    
    for i in range(1024):
        prob_list[i]/=prob_sum

    datas=np.random.choice(a=range(1024), size=100000, p=prob_list)
    
    maxiter=100
    qnn.fit(datas, maxiter)
    
    data_param=qnn.predict()

    for i in range(512):
        assert abs(data_param[i]-prob_list[i])<0.1
        
def test_bar_stripe():

    n_qubit = 12
    depth = 6
    time_step = 0.5
    circuit = create_farhi_neven_ansatz(n_qubit, depth)
    qnn = QNNGeneretor(circuit,"same",0,9)

    #5000個のデータを作る
    datas=[]
    
    for _ in range(5000):
        kazu=0
        if random.random() < 0.5:
            kazu = int(random.random()*8) * (1+8+64)
        else:
            if random.random() < 0.5:
                kazu += 7
            if random.random() < 0.5:
                kazu += 7*8
            if random.random() < 0.5:
                kazu += 7*64
        datas.append(kazu)

    maxiter=100
    qnn.fit(datas, maxiter)
    
    data_param=qnn.predict()
    risou_kak = np.zeros(512)
    for i in range(8):
        risou_kak[i*73]+=0.0625
        uuu=0
        if (i&4)>0:
           uuu+=64
        if (i&2)>0:
           uuu+=8
        if (i&1)>0:
           uuu+=1
        risou_kak[uuu*7]+=0.0625

    for i in range(512):
        assert abs(data_param[i]-risou_kak[i])<0.1

def test_bar_stripe_hamming():

    n_qubit = 12
    depth = 6
    time_step = 0.5
    circuit = create_farhi_neven_ansatz(n_qubit, depth)
    qnn = QNNGeneretor(circuit,"exp_hamming",0.07,9)

    #5000個のデータを作る
    datas=[]
    
    for _ in range(5000):
        kazu=0
        if random.random() < 0.5:
            kazu = int(random.random()*8) * (1+8+64)
        else:
            if random.random() < 0.5:
                kazu += 7
            if random.random() < 0.5:
                kazu += 7*8
            if random.random() < 0.5:
                kazu += 7*64
        datas.append(kazu)

    maxiter=100
    qnn.fit(datas, maxiter)
    
    data_param=qnn.predict()
    risou_kak = np.zeros(512)
    for i in range(8):
        risou_kak[i*73]+=0.0625
        uuu=0
        if (i&4)>0:
           uuu+=64
        if (i&2)>0:
           uuu+=8
        if (i&1)>0:
           uuu+=1
        risou_kak[uuu*7]+=0.0625

    for i in range(512):
        assert abs(data_param[i]-risou_kak[i])<0.1

