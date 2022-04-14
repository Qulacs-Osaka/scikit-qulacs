from math import exp

import numpy as np

from skqulacs.circuit import create_farhi_neven_ansatz
from skqulacs.qnn import QNNGeneretor
n_qubit = 10
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

maxiter = 15
qnn.fit(datas, maxiter)

data_param = qnn.predict()

gosa = 0
for i in range(3, 61):
    hei = 0
    for j in range(-3, 3 + 1):
        hei += (data_param[i + j] - prob_list[i + j]) / 7
    gosa += abs(hei)
assert gosa / 2 < 0.08