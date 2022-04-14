from math import exp

import numpy as np

from skqulacs.circuit import create_farhi_neven_ansatz
from skqulacs.qnn import QNNGeneretor


def test_mix_gauss_mini():

    n_qubit = 4
    depth = 10
    circuit = create_farhi_neven_ansatz(n_qubit, depth)
    qnn = QNNGeneretor(circuit, "gauss", 0.2, 2)

    # 100000個のデータを作る
    prob_list = np.zeros(4)
    prob_list[0] = 0.1
    prob_list[1] = 0.2
    prob_list[2] = 0.3
    prob_list[3] = 0.4

    maxiter = 5
    qnn.fit_direct_distribution(prob_list, maxiter)

    data_param = qnn.predict()

    for i in range(4):
        print(data_param[i], prob_list[i], data_param[i] - prob_list[i])
        assert abs(data_param[i] - prob_list[i]) < 0.04


def test_mix_gauss():

    n_qubit = 6
    depth = 10
    circuit = create_farhi_neven_ansatz(n_qubit, depth)
    qnn = QNNGeneretor(circuit, "gauss", 4, 6)

    # 100000個のデータを作る
    prob_list = np.zeros(64)
    ua = 64 * 2 / 7
    ub = 64 * 5 / 7
    v = 64 * 1 / 8
    prob_sum = 0
    for i in range(64):
        prob_list[i] = exp(-(ua - i) * (ua - i) / (2 * v * v)) + exp(
            -(ub - i) * (ub - i) / (2 * v * v)
        )
        prob_sum += prob_list[i]

    for i in range(64):
        prob_list[i] /= prob_sum

    datas = np.random.choice(a=range(64), size=10000, p=prob_list)

    maxiter = 120
    qnn.fit(datas, maxiter)

    data_param = qnn.predict()

    gosa = 0
    for i in range(3, 61):
        hei = 0
        for j in range(-3, 3 + 1):
            hei += (data_param[i + j] - prob_list[i + j]) / 7
        gosa += abs(hei)
    assert gosa / 2 < 0.08


"""
test_bar_stripeは短時間でまともな成果を得られないのでテストから外された
80%程度の正解率は出る。

def test_bar_stripe():

    n_qubit = 9
    depth = 22
    circuit = create_farhi_neven_ansatz(n_qubit, depth)
    qnn = QNNGeneretor(circuit, "same", 0, 9)

    prob_list = np.zeros(512)
    for i in range(8):
        prob_list[i * 73] += 0.0625
        uuu = 0
        if (i & 4) > 0:
            uuu += 64
        if (i & 2) > 0:
            uuu += 8
        if (i & 1) > 0:
            uuu += 1
        prob_list[uuu * 7] += 0.0625

    datas = np.random.choice(a=range(512), size=100000, p=prob_list)

    maxiter = 600
    qnn.fit(datas, maxiter)

    data_param = qnn.predict()

    gosa = 0
    for i in range(512):
        gosa += abs(data_param[i] - prob_list[i])
        if prob_list[i] > 0.001:
            print(i, data_param[i])
    assert gosa < 0.4


def test_bar_stripe_hamming():

    n_qubit = 9
    depth = 13
    circuit = create_farhi_neven_ansatz(n_qubit, depth)
    qnn = QNNGeneretor(circuit,"exp_hamming",0.07,9)

    prob_list = np.zeros(512)
    for i in range(8):
        prob_list[i*73]+=0.0625
        uuu=0
        if (i&4)>0:
           uuu+=64
        if (i&2)>0:
           uuu+=8
        if (i&1)>0:
           uuu+=1
        prob_list[uuu*7]+=0.0625
    datas=np.random.choice(a=range(512), size=100000, p=prob_list)
    maxiter=500
    qnn.fit(datas, maxiter)
    data_param=qnn.predict()
    gosa=0
    for i in range(512):
        gosa+=abs(data_param[i]-prob_list[i])
        if(prob_list[i]>0.001):
            print(i,data_param[i])
    assert gosa<0.2
"""
