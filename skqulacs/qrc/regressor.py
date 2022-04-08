from __future__ import annotations

import random
from typing import List, Optional

import numpy as np
from qulacs import Observable, QuantumState
from qulacs.gate import RX, RZ
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from skqulacs.circuit import LearningCircuit
from skqulacs.circuit.pre_defined import create_qcl_ansatz


class QRCRegressor:
    def __init__(
        self,
        n_qubit: int,
        circuit: LearningCircuit,
        observables: List[Observable],
    ) -> None:
        self.n_qubit = n_qubit
        self.circuit = circuit
        self.observables = observables

    def fit(
        self,
        x_train: List[List[float]],
        y_train: List[float],
        maxiter: Optional[int] = None,
        generate_observables: bool = True,
        generate_circuit: bool = True,
        circuit_depth: int = 5,
        n_qubit: int = 8,
    ) -> None:
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        self.x_scaler.fit(x_train)
        self.y_scaler.fit(y_train)
        x_train_scaled, y_train_scaled = self.x_scaler(x_train), self.y_scaler(y_train)

        if generate_observables:
            self.observables = self.create_observables()
        if generate_circuit:
            self.circuit = self.create_random_circuit(n_qubit, circuit_depth)

        observation_results = self.get_observation_results(x_train_scaled)
        self.regression = LinearRegression()
        self.regression.fit(observation_results, y_train_scaled)

    def predict(self, x_test: List[List[float]]) -> List[float]:
        x_test_scaled = self.x_scaler.transform(x_test)
        observation_results = self.get_observation_results(x_test_scaled)
        ret_val: List[float] = self.y_scaler.inverse_transform(
            self.regression.predict(observation_results)
        )
        return ret_val

    def score(self, x_test: List[List[float]], y_test: List[List[float]]) -> float:
        x_test_scaled, y_test_scaled = self.x_scaler(x_test), self.y_scaler(y_test)
        observation_results = self.get_observation_results(x_test_scaled)
        ret_val: float = self.regression.score(observation_results, y_test_scaled)
        return ret_val

    def create_observables(self) -> List[Observable]:
        observables = list()
        for _ in range(80):
            observable = Observable(self.n_qubit)
            observable.add_random_operator(random.randint(2, 10))
            observables.append(observable)
        return observables

    def get_observation_results(self, X: List[List[float]]) -> List[float]:
        observation_results: List[float] = list()

        for x in X:
            state = QuantumState(self.n_qubit)
            state.set_zero_state()
            for i in range(len(x)):
                if (i // self.n_qubit) % 2 == 0:
                    RX(i % self.n_qubit, x[i] * np.pi).update_quantum_state(state)
                else:
                    RZ(i % self.n_qubit, x[i] * np.pi).update_quantum_state(state)

            self.circuit._circuit.update_quantum_state(state)

            observation_results += [
                observable.get_expectation_value(state)
                for observable in self.observables
            ]

        return observation_results

    def create_random_circuit(self, n_qubit: int, c_depth: int) -> LearningCircuit:
        circuit = create_qcl_ansatz(n_qubit, c_depth, seed=0)
        return circuit
