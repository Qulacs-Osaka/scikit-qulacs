from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional
from qulacs import QuantumState, ParametricQuantumCircuit


class _Axis(Enum):
    X = auto()
    Y = auto()
    Z = auto()


@dataclass
class _InputParameter:
    position: int
    func: Optional[Callable[[float], float]] = field(default=None)


@dataclass
class _Parameter:
    position: int
    parameter: float


class LearningCircuit:
    def __init__(
        self,
        n_qubit: int,
        circuit_depth: int,
        time_step: float,
    ) -> None:
        self.n_qubit = n_qubit
        self.circuit_depth = circuit_depth
        self.time_step = time_step
        self.circuit = ParametricQuantumCircuit(n_qubit)
        self.parameter_list: List[_Parameter] = []
        self.input_parameter_list: List[_InputParameter] = []
        self.current_gate_position = 0
        self.current_input_gate_position = 0

    def add_X_gate(self, index: int):
        self.circuit.add_X_gate(index)

    def add_Y_gate(self, index: int):
        self.circuit.add_Y_gate(index)

    def add_Z_gate(self, index: int):
        self.circuit.add_Z_gate(index)

    def _add_R_gate_inner(
        self,
        index: int,
        parameter: Optional[float],
        target: _Axis,
    ):
        if target == _Axis.X:
            self.circuit.add_RX_gate(index, parameter)
        # elif target == _Axis.Y:
        #     self.circuit.add_parametric_RY_gate(index, parameter)
        # elif target == _Axis.Z:
        #     self.circuit.add_parametric_RZ_gate(index, parameter)
        else:
            raise NotImplementedError
        self.current_gate_position += 1

    def add_RX_gate(
        self, index: int, parameter: float
    ):
        self._add_R_gate_inner(index, parameter, _Axis.X)

    # def add_RY_gate(
    #     self, index: int, parameter: float, input_func: Callable[[float], float]
    # ):
    #     self._add_R_gate_inner(index, parameter, _Axis.Y, input_func)

    # def add_RZ_gate(
    #     self, index: int, parameter: float, input_func: Callable[[float], float]
    # ):
    #     self._add_R_gate_inner(index, parameter, _Axis.Z, input_func)

    def _add_input_R_gate_inner(
        self,
        index: int,
        target: _Axis,
        input_func: Callable[[float], float] = lambda x: x,
    ):
        if target == _Axis.X:
            self.circuit.add_parametric_RX_gate(index, 0.0)
            # elif target == _Axis.Y:
            #     self.circuit.add_parametric_RY_gate(index, parameter)
            # elif target == _Axis.Z:
            #     self.circuit.add_parametric_RZ_gate(index, parameter)
            # else:
            # raise NotImplementedError
        self.input_parameter_list.append(_InputParameter(
            self.current_input_gate_position, input_func))
        self.current_input_gate_position += 1

    def add_input_RX_gate(
        self,
        index: int,
        # parameter: Optional[float] = None,
        input_func: Callable[[float], float] = lambda x: x,
    ):
        self._add_input_R_gate_inner(
            index, _Axis.X, input_func
        )

    def _add_parametric_R_gate_inner(
        self,
        index: int,
        parameter: float,
        target: _Axis,
        # input_func: Callable[[float], float],
    ):
        if target == _Axis.X:
            self.circuit.add_parametric_RX_gate(index, parameter)
        elif target == _Axis.Y:
            self.circuit.add_parametric_RY_gate(index, parameter)
        elif target == _Axis.Z:
            self.circuit.add_parametric_RZ_gate(index, parameter)
        else:
            raise NotImplementedError
        self.parameter_list.append(_Parameter(self.current_gate_position, parameter))
        self.current_gate_position += 1

    def add_parametric_RX_gate(
        self, index: int, parameter: float
    ):
        self._add_parametric_R_gate_inner(
            index, parameter, _Axis.X)

    # def add_parametric_RY_gate(
    #     self, index: int, parameter: float, input_func: Callable[[float], float]
    # ):
    #     self._add_parametric_R_gate_inner(
    #         index, parameter, _Axis.Y, input_func)

    # def add_parametric_RZ_gate(
    #     self, index: int, parameter: float, input_func: Callable[[float], float]
    # ):
    #     self._add_parametric_R_gate_inner(
    #         index, parameter, _Axis.Z, input_func)

    def update_noninput_parameter(self, theta: List[float]):
        current_theta_index = 0
        for parameter in self.parameter_list:
            self.circuit.set_parameter(
                parameter.position, theta[current_theta_index])
            current_theta_index += 1

    def get_parameter(self):
        parameter_count = self.circuit.get_parameter_count()
        theta = [self.circuit.get_parameter(i) for i in range(parameter_count)]
        return theta

    def run(self, x: float) -> QuantumState:
        state = QuantumState(self.n_qubit)
        state.set_zero_state()
        for parameter in self.input_parameter_list:
            self.circuit.set_parameter(
                parameter.position, parameter.func(x)
            )
        self.circuit.update_quantum_state(state)
        return state
