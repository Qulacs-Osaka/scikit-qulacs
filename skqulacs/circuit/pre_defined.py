from functools import reduce
from math import factorial
from typing import List, Optional

import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import NDArray
from qulacs.gate import CZ, DenseMatrix

from .circuit import LearningCircuit


def create_qcl_ansatz(
    n_qubit: int, c_depth: int, time_step: float = 0.5, seed: Optional[int] = None
) -> LearningCircuit:
    """Create a circuit used in this page: https://dojo.qulacs.org/ja/latest/notebooks/5.2_Quantum_Circuit_Learning.html
    Args:
        n_qubit: number of qubits
        c_depth: circuit depth
        time_step: the evolution time used for the hamiltonian dynamics
        seed: seed for random numbers. used for determining the interaction strength of the hamiltonian simulation
    Examples:
        >>> n_qubit = 4
        >>> circuit = create_qcl_ansatz(n_qubit, 3, 0.5)
        >>> qnn = QNNRegressor(circuit)
        >>> qnn.fit(x_train, y_train)
    """

    def preprocess_x(x: NDArray[np.float_], index: int) -> float:
        xa = x[index % len(x)]
        return min(1, max(-1, xa))

    circuit = LearningCircuit(n_qubit)
    for i in range(n_qubit):
        # Capture copy of i by `i=i`.
        # Without this, i in lambda is a reference to the i, so the lambda always
        # recognize i as n_qubit - 1.
        circuit.add_input_RY_gate(i, lambda x, i=i: np.arcsin(preprocess_x(x, i)))
        circuit.add_input_RZ_gate(
            i, lambda x, i=i: np.arccos(preprocess_x(x, i) * preprocess_x(x, i))
        )

    rng = default_rng(seed)
    time_evol_gate = _create_time_evol_gate(n_qubit, time_step)
    for _ in range(c_depth):
        circuit.add_gate(time_evol_gate)
        for i in range(n_qubit):
            angle = 2.0 * np.pi * rng.random()
            circuit.add_parametric_RX_gate(i, angle)
            angle = 2.0 * np.pi * rng.random()
            circuit.add_parametric_RZ_gate(i, angle)
            angle = 2.0 * np.pi * rng.random()
            circuit.add_parametric_RX_gate(i, angle)
    return circuit


def _create_time_evol_gate(
    n_qubit, time_step=0.77, rng: Generator = None, seed: int = 0
):
    """create a hamiltonian dynamics with transverse field ising model with random interaction and random magnetic field
    Args:
        n_qubit: number of qubits
        time_step: evolution time
        rng: random number generator
        seed: seed for random number
    Return:
        qulacs' gate object
    """
    if rng is None:
        rng = default_rng(seed)

    ham = _make_hamiltonian(n_qubit, rng)
    # Create time evolution operator by diagonalization.
    # H*P = P*D <-> H = P*D*P^dagger
    diag, eigen_vecs = np.linalg.eigh(ham)
    time_evol_op = np.dot(
        np.dot(eigen_vecs, np.diag(np.exp(-1j * time_step * diag))), eigen_vecs.T.conj()
    )  # e^-iHT

    # Convert to a qulacs gate
    time_evol_gate = DenseMatrix([i for i in range(n_qubit)], time_evol_op)

    return time_evol_gate


def _make_hamiltonian(n_qubit, rng: Generator = None, seed: int = 0):
    if rng is None:
        rng = default_rng(seed)
    X_mat = np.array([[0, 1], [1, 0]])
    Z_mat = np.array([[1, 0], [0, -1]])
    ham = np.zeros((2**n_qubit, 2**n_qubit), dtype=complex)
    for i in range(n_qubit):
        Jx = rng.uniform(-1.0, 1.0)
        ham += Jx * _make_fullgate([[i, X_mat]], n_qubit)
        for j in range(i + 1, n_qubit):
            J_ij = rng.uniform(-1.0, 1.0)
            ham += J_ij * _make_fullgate([[i, Z_mat], [j, Z_mat]], n_qubit)
    return ham


def _make_fullgate(list_SiteAndOperator, n_qubit):
    """
    Receive `list_SiteAndOperator = [ [i_0, O_0], [i_1, O_1], ...]` and
    insert identity to qubits which is not present in the list to create (2**n_qubit, 2**n_qubit) matrix
    I(0) * ... * O_0(i_0) * ... * O_1(i_1) ...
    """
    I_mat = np.eye(2, dtype=complex)
    list_Site = [SiteAndOperator[0] for SiteAndOperator in list_SiteAndOperator]
    list_SingleGates = []
    cnt = 0
    for i in range(n_qubit):
        if i in list_Site:
            list_SingleGates.append(list_SiteAndOperator[cnt][1])
            cnt += 1
        else:
            list_SingleGates.append(I_mat)
    return reduce(np.kron, list_SingleGates)


def create_farhi_neven_ansatz(
    n_qubit: int, c_depth: int, seed: Optional[int] = None
) -> LearningCircuit:
    """create circuit proposed in https://arxiv.org/abs/1802.06002.
    Args:
        n_qubits: number of qubits
        c_depth: depth of the circuit
        seed: random seed determining the shuffling of the qubits between layers
    """

    def preprocess_x(x: NDArray[np.float_], index: int):
        xa = x[index % len(x)]
        return min(1, max(-1, xa))

    circuit = LearningCircuit(n_qubit)
    for i in range(n_qubit):
        circuit.add_input_RY_gate(i, lambda x, i=i: np.arcsin(preprocess_x(x, i)))
        circuit.add_input_RZ_gate(
            i, lambda x, i=i: np.arccos(preprocess_x(x, i) * preprocess_x(x, i))
        )

    zyu = list(range(n_qubit))
    rng = default_rng(seed)
    for _ in range(c_depth):
        rng.shuffle(zyu)

        for i in range(0, n_qubit - 1, 2):
            angle_x = 2.0 * np.pi * rng.random()
            angle_y = 2.0 * np.pi * rng.random()
            circuit.add_CNOT_gate(zyu[i + 1], zyu[i])
            circuit.add_parametric_RX_gate(zyu[i], angle_x)
            circuit.add_parametric_RY_gate(zyu[i], angle_y)
            circuit.add_CNOT_gate(zyu[i + 1], zyu[i])
            angle_x = 2.0 * np.pi * rng.random()
            angle_y = 2.0 * np.pi * rng.random()
            circuit.add_parametric_RY_gate(zyu[i], -angle_y)
            circuit.add_parametric_RX_gate(zyu[i], -angle_x)
    return circuit


def create_farhi_neven_watle_ansatz(
    n_qubit: int, c_depth: int, seed: Optional[int] = None
) -> LearningCircuit:
    """create modified version of circuit proposed in https://arxiv.org/abs/1802.06002.
    made by WA_TLE.
    Args:
        n_qubits: number of qubits
        c_depth: depth of the circuit
        seed: random seed determining the shuffling of the qubits between layers
    """
    xkeisuu = np.zeros([25, 25, 25])
    nCr = np.zeros([25, 25])
    for i in range(25):
        for j in range(i + 1):
            nCr[i][j] = factorial(i) / factorial(j) / factorial(i - j)
    for i in range(25):
        for j in range(i):
            if j == 0:
                xkeisuu[i][0][i] = 1
            else:
                for k in range(i - j, i + 1):
                    xkeisuu[i][j][k] = xkeisuu[i][j - 1][k] + nCr[i][j] * nCr[j][
                        i - k
                    ] * ((-1) ** (i + j + k))

    def preprocess_x(x: NDArray[np.float_], index: int):
        dex = index % len(x)
        qubits_per_bit = ((n_qubit - dex) - 1) // len(x) + 1
        xa = (min(1, max(-1, x[dex])) + 1) / 2
        sban = index // len(x)

        xb = 0
        if qubits_per_bit < 25:
            for i in range(qubits_per_bit):
                xb += xkeisuu[qubits_per_bit][sban][qubits_per_bit - i]
                xb *= xa
        else:
            # Overflow `double` type of Combination(n, r).
            # There are few cases to use the same 25 bits.
            xb = xa

        if xb < 0 or 1 < xb:
            raise RuntimeError("bug")

        return xb * 2 - 1

    circuit = LearningCircuit(n_qubit)
    for i in range(n_qubit):
        circuit.add_input_RY_gate(i, lambda x, i=i: np.arcsin(preprocess_x(x, i)))
        circuit.add_input_RZ_gate(
            i, lambda x, i=i: np.arccos(preprocess_x(x, i) * preprocess_x(x, i))
        )

    zyu = list(range(n_qubit))
    rng = default_rng(seed)
    for _ in range(c_depth):
        rng.shuffle(zyu)
        for i in range(0, n_qubit - 1, 2):
            angle_x = 2.0 * np.pi * rng.random()
            angle_y = 2.0 * np.pi * rng.random()
            circuit.add_CNOT_gate(zyu[i + 1], zyu[i])
            circuit.add_parametric_RX_gate(zyu[i], angle_x)
            circuit.add_parametric_RY_gate(zyu[i], angle_y)
            circuit.add_CNOT_gate(zyu[i + 1], zyu[i])
            angle_x = 2.0 * np.pi * rng.random()
            angle_y = 2.0 * np.pi * rng.random()
            circuit.add_parametric_RY_gate(zyu[i], -angle_y)
            circuit.add_parametric_RX_gate(zyu[i], -angle_x)
    return circuit


def create_ibm_embedding_circuit(n_qubit: int) -> LearningCircuit:
    """create circuit proposed in https://arxiv.org/abs/1802.06002.
    Args:
        n_qubits: number of qubits
    """

    def preprocess_x(x: NDArray[np.float_], index: int) -> float:
        xa = x[index % len(x)]
        return xa

    circuit = LearningCircuit(n_qubit)
    for i in range(n_qubit):
        circuit.add_H_gate(i)

    for i in range(n_qubit):
        j = (i + 1) % n_qubit
        circuit.add_input_RZ_gate(i, lambda x, i=i: preprocess_x(x, i))
        circuit.add_CNOT_gate(i, j)
        circuit.add_input_RZ_gate(
            j,
            lambda x, i=i: (
                (np.pi - preprocess_x(x, i)) * (np.pi - preprocess_x(x, j))
            ),
        )
        circuit.add_CNOT_gate(i, j)

    for i in range(n_qubit):
        circuit.add_H_gate(i)

    for i in range(n_qubit):
        j = (i + 1) % n_qubit
        circuit.add_input_RZ_gate(i, lambda x, i=i: preprocess_x(x, i))
        circuit.add_CNOT_gate(i, j)
        circuit.add_input_RZ_gate(
            j,
            lambda x, i=i: (
                (np.pi - preprocess_x(x, i)) * (np.pi - preprocess_x(x, j))
            ),
        )
        circuit.add_CNOT_gate(i, j)
    return circuit


def create_shirai_ansatz(
    n_qubit: int, c_depth: int = 5, seed: int = 0
) -> LearningCircuit:
    """create circuit proposed in http://arxiv.org/abs/2111.02951.
    Args:
        n_qubit: number of qubits
        c_depth: circuit depth as defined in http://arxiv.org/abs/2111.02951
        seed: random seed for initial parameter values
    """

    def preprocess_x(x: NDArray[np.float_], index: int) -> float:
        xa = x[index % len(x)]
        return xa

    rng = default_rng(seed)
    circuit = LearningCircuit(n_qubit)
    for c_kai in range(c_depth):
        # input embedding layer
        for i in range(n_qubit):
            circuit.add_input_RZ_gate(i, lambda x, i=i: np.arcsin(preprocess_x(x, i)))
            for j in range(i):

                circuit.add_CNOT_gate(j, i)
                circuit.add_input_RZ_gate(
                    i,
                    lambda x, i=i: -np.arcsin(preprocess_x(x, i) * preprocess_x(x, j))
                    / 2,
                )
                circuit.add_CNOT_gate(j, i)
                circuit.add_input_RZ_gate(
                    i,
                    lambda x, i=i: np.arcsin(preprocess_x(x, i) * preprocess_x(x, j))
                    / 2,
                )
        # trainable layer
        for i in range(0, n_qubit):
            j = (i + 1) % n_qubit

            angle = 2.0 * np.pi * rng.random()
            circuit.add_parametric_RX_gate(i, angle)
            angle = 2.0 * np.pi * rng.random()
            circuit.add_parametric_RZ_gate(i, angle)
            angle = 2.0 * np.pi * rng.random()
            circuit.add_parametric_RX_gate(i, angle)

            circuit.add_CNOT_gate(i, j)
            angle = 2.0 * np.pi * rng.random()
            circuit.add_parametric_RX_gate(i, angle)
            angle = 2.0 * np.pi * rng.random()
            circuit.add_parametric_RZ_gate(i, angle)
            angle = 2.0 * np.pi * rng.random()
            circuit.add_parametric_RX_gate(i, angle)
            angle = 2.0 * np.pi * rng.random()
            circuit.add_parametric_RX_gate(j, angle)
            angle = 2.0 * np.pi * rng.random()
            circuit.add_parametric_RZ_gate(j, angle)
            angle = 2.0 * np.pi * rng.random()
            circuit.add_parametric_RX_gate(j, angle)
            circuit.add_CNOT_gate(i, j)
            angle = 2.0 * np.pi * rng.random()
            circuit.add_parametric_RX_gate(i, angle)
            angle = 2.0 * np.pi * rng.random()
            circuit.add_parametric_RZ_gate(i, angle)
            angle = 2.0 * np.pi * rng.random()
            circuit.add_parametric_RX_gate(i, angle)
            angle = 2.0 * np.pi * rng.random()
            circuit.add_parametric_RX_gate(j, angle)
            angle = 2.0 * np.pi * rng.random()
            circuit.add_parametric_RZ_gate(j, angle)
            angle = 2.0 * np.pi * rng.random()
            circuit.add_parametric_RX_gate(j, angle)

    return circuit


def create_npqc_ansatz(
    n_qubit: int, c_depth: int = 4, c: float = 0.1
) -> LearningCircuit:
    """
    Creates circuit used in http://arxiv.org/abs/2108.01039, Fig. 5(a).
    Args:
        n_qubit: number of qubits. must be even.
        c_depth: circuit depth. The number of parameters is 8+4*(c_depth-1).
        c: hyperparameter of the circuit. Defined in Eq. (2) of the paper.
    """

    if n_qubit % 2 != 0:
        raise ValueError(
            "create_large_qsv takes only integer number of qubits, but given "
            + str(n_qubit)
        )

    def preprocess_x(x: NDArray[np.float_], index: int) -> float:
        xa = x[index % len(x)]
        return xa

    circuit = LearningCircuit(n_qubit)
    ban = 0
    for i in range(n_qubit):
        circuit.add_input_RY_gate(
            i, lambda x, ban_lam=ban: preprocess_x(x, ban_lam) * c + np.pi / 2
        )
        ban = ban + 1
        circuit.add_input_RZ_gate(
            i, lambda x, ban_lam=ban: preprocess_x(x, ban_lam) * c + np.pi / 2
        )
        ban = ban + 1

    for c_kai in range(c_depth):
        for i in range(0, n_qubit - 1, 2):
            circuit.add_RY_gate(i, np.pi / 2)
            recC = c_kai + 1
            recA = 0
            while recC % 2 == 0:
                recC /= 2
                recA += 1
            circuit.add_gate(CZ(i, (i + recA * 2 + 1) % n_qubit))
            circuit.add_input_RY_gate(
                i, lambda x, ban_lam=ban: preprocess_x(x, ban_lam) * c + np.pi / 2
            )
            ban = ban + 1
            if c_kai + 1 < c_depth:
                circuit.add_input_RZ_gate(
                    i,
                    lambda x, ban_lam=ban: preprocess_x(x, ban_lam) * c + np.pi / 2,
                )
                ban = ban + 1
    return circuit


def create_yzcx_ansatz(
    n_qubit: int, c_depth: int = 4, c: float = 0.1, seed: int = 9
) -> LearningCircuit:
    """
    Creates circuit used in http://arxiv.org/abs/2108.01039, Fig. 5(c).
    Args:
        n_qubit: number of qubits. must be even.
        c_depth: circuit depth. The number of parameters is 8*c_depth.
        c: hyperparameter of the circuit. Defined in Eq. (2) of the paper.
    """

    def preprocess_x(x: NDArray[np.float_], index: int) -> float:
        xa = x[index % len(x)]
        return xa

    rng = default_rng(seed)
    circuit = LearningCircuit(n_qubit)
    ban = 0
    for c_kai in range(c_depth):
        for i in range(0, n_qubit):
            angle = 2.0 * np.pi * rng.random()
            circuit.add_input_RY_gate(
                i, lambda x, ban_lam=ban: preprocess_x(x, ban_lam) * c
            )
            circuit.add_parametric_RY_gate(i, angle)
            ban = ban + 1
            angle = 2.0 * np.pi * rng.random()
            circuit.add_input_RZ_gate(
                i, lambda x, ban_lam=ban: preprocess_x(x, ban_lam) * c
            )
            circuit.add_parametric_RZ_gate(i, angle)
            ban = ban + 1
            if i % 2 == c_kai % 2 and i + 1 < n_qubit:
                circuit.add_CNOT_gate(i, i + 1)
    return circuit


def create_qcnn_ansatz(n_qubit: int, seed: int = 0) -> LearningCircuit:
    """
    Creates circuit used in https://www.tensorflow.org/quantum/tutorials/qcnn?hl=en, Section 1.
    Args:
        n_qubit: number of qubits. must be even.
        seed: seed for random numbers. used for determining the interaction strength of the hamiltonian simulation
    """

    rng = default_rng(seed)

    # def _innser_conv_circuit1(circuit: LearningCircuit, src: int, dest: int):
    #     angle = rng.uniform(-np.pi, np.pi)
    #     circuit.add_parametric_RY_gate(src, angle)
    #     angle = rng.uniform(-np.pi, np.pi)
    #     circuit.add_parametric_RY_gate(dest, angle)
    #     circuit.add_CNOT_gate(src, dest)
    #     return circuit

    def one_qubit_unitary(circuit: LearningCircuit, index: int):
        ids = []
        angle = rng.uniform(-np.pi, np.pi)
        circuit.add_parametric_RX_gate(index, angle)
        ids.append(circuit.get_parameter_count())
        angle = rng.uniform(-np.pi, np.pi)
        circuit.add_parametric_RY_gate(index, angle)
        ids.append(circuit.get_parameter_count())
        angle = rng.uniform(-np.pi, np.pi)
        circuit.add_parametric_RZ_gate(index, angle)
        ids.append(circuit.get_parameter_count())
        return ids

    def two_qubit_unitary(
        circuit: LearningCircuit, target: List[int], pauli_ids: List[int]
    ):
        angle = rng.uniform(-np.pi, np.pi)
        circuit.add_parametric_multi_Pauli_rotation_gate(target, pauli_ids, angle)

    def _innser_conv_circuit2(circuit: LearningCircuit, src: int, dest: int):
        one_qubit_unitary(circuit, src)
        one_qubit_unitary(circuit, dest)

        target = [src, dest]
        pauli_xx_ids = [1, 1]
        two_qubit_unitary(circuit, target, pauli_xx_ids)
        pauli_yy_ids = [2, 2]
        two_qubit_unitary(circuit, target, pauli_yy_ids)
        pauli_zz_ids = [3, 3]
        two_qubit_unitary(circuit, target, pauli_zz_ids)

        one_qubit_unitary(circuit, src)
        one_qubit_unitary(circuit, dest)
        return circuit

    def conv_circuit(circuit: LearningCircuit, src: int, dest: int):
        # return _innser_conv_circuit1(circuit, src, dest)
        return _innser_conv_circuit2(circuit, src, dest)

    # def pooling_circuit(circuit: LearningCircuit, src: int, dest: int):
    #     angle = rng.uniform(-np.pi, np.pi)
    #     circuit.add_parametric_RZ_gate(src, angle)
    #     circuit.add_CNOT_gate(src, dest)
    #     angle = rng.uniform(-np.pi, np.pi)
    #     circuit.add_parametric_RX_gate(dest, angle)
    #     circuit.add_X_gate(src)
    #     circuit.add_CNOT_gate(src, dest)
    #     return circuit

    def pooling_circuit(circuit: LearningCircuit, src: int, dest: int):
        ids = one_qubit_unitary(circuit, dest)
        one_qubit_unitary(circuit, src)
        circuit.add_CNOT_gate(src, dest)
        angle = rng.uniform(-np.pi, np.pi)
        circuit.add_parametric_RZ_gate(dest, angle, share_with=ids[2])
        circuit.add_parametric_RY_gate(dest, angle, share_with=ids[1])
        circuit.add_parametric_RX_gate(dest, angle, share_with=ids[0])
        return circuit

    circuit = LearningCircuit(n_qubit)
    for i in range(n_qubit):
        circuit.add_input_RX_gate(i, lambda x: x)

    # cluster state
    for i in range(n_qubit):
        circuit.add_H_gate(i)
    for this_bit in range(n_qubit):
        next_bit = this_bit + 1 if this_bit < n_qubit - 1 else 0
        # print(f"this_bit: {this_bit} next_bit: {next_bit}")
        circuit.add_CNOT_gate(this_bit, next_bit)
        circuit.add_Z_gate(next_bit)

    # depth 1 (0, 1)
    circuit = conv_circuit(circuit, 0, 1)
    circuit = pooling_circuit(circuit, 0, 1)
    circuit = conv_circuit(circuit, 2, 3)
    circuit = pooling_circuit(circuit, 2, 3)
    circuit = conv_circuit(circuit, 4, 5)
    circuit = pooling_circuit(circuit, 4, 5)
    circuit = conv_circuit(circuit, 6, 7)
    circuit = pooling_circuit(circuit, 6, 7)

    # depth 2 (1, 3)
    circuit = conv_circuit(circuit, 1, 3)
    circuit = pooling_circuit(circuit, 1, 3)
    circuit = conv_circuit(circuit, 5, 7)
    circuit = pooling_circuit(circuit, 5, 7)

    # depth 3 (3, 7)
    circuit = conv_circuit(circuit, 3, 7)
    circuit = pooling_circuit(circuit, 3, 7)

    # # 二分木を作成してペアを作る
    # # 枝から幅優先で二分木を作らないと精度が悪くなる。。
    # targets = []

    # def tree(ns):
    #     n = len(ns)
    #     if n <= 0:
    #         return
    #     node = {}
    #     node["ns"] = ns
    #     left = tree(ns[:n // 2])
    #     right = tree(ns[n - (n // 2):])
    #     if left is not None and right is not None:
    #         targets.append([max(left["ns"]), max(right["ns"])])
    #     return node

    # tree([x for x in range(n_qubit)])
    # for t in targets:
    #     circuit = conv_circuit(circuit, t[0], t[1])
    #     circuit = pooling_circuit(circuit, t[0], t[1])

    return circuit
