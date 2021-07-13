from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional, Union
from qulacs import QuantumState, ParametricQuantumCircuit


class _Axis(Enum):
    X = auto()
    Y = auto()
    Z = auto()


@dataclass
class _Parameter:
    pos: int
    theta_pos: Optional[int]  # position at theta list.
    value: Optional[float]
    func: Union[
        Callable[[float], float], Callable[[float, float], float], None
    ] = field(default=None)

    def is_parametric(self) -> bool:
        """Return if this parameter is for learning."""
        return self.value is not None and self.theta_pos is not None

    def is_input(self) -> bool:
        return self.func is not None

    def set_parameter(self, parameter: float):
        self.value = parameter


class LearningCircuit:
    def __init__(
        self,
        n_qubit: int,
        time_step: float,
    ) -> None:
        self.n_qubit = n_qubit
        self.time_step = time_step
        self.circuit = ParametricQuantumCircuit(n_qubit)
        self.parameter_list: List[_Parameter] = []
        self.parametric_gate_count = 0
        self.learning_gate_count = 0

    def add_gate(self, gate):
        self.circuit.add_gate(gate)

    def add_X_gate(self, index: int):
        self.circuit.add_X_gate(index)

    def add_Y_gate(self, index: int):
        self.circuit.add_Y_gate(index)

    def add_Z_gate(self, index: int):
        self.circuit.add_Z_gate(index)

    def _add_R_gate_inner(
        self,
        index: int,
        angle: Optional[float],
        target: _Axis,
    ):
        if target == _Axis.X:
            self.circuit.add_RX_gate(index, angle)
        elif target == _Axis.Y:
            self.circuit.add_parametric_RY_gate(index, angle)
        elif target == _Axis.Z:
            self.circuit.add_parametric_RZ_gate(index, angle)
        else:
            raise NotImplementedError

    def add_RX_gate(self, index: int, angle: float):
        self._add_R_gate_inner(index, angle, _Axis.X)

    def add_RY_gate(self, index: int, parameter: float):
        self._add_R_gate_inner(index, parameter, _Axis.Y)

    def add_RZ_gate(self, index: int, parameter: float):
        self._add_R_gate_inner(index, parameter, _Axis.Z)

    def _add_parametric_R_gate_inner(
        self,
        index: int,
        parameter: float,
        target: _Axis,
        # input_func: Callable[[float], float],
    ):
        self.parameter_list.append(
            _Parameter(
                self.circuit.get_parameter_count(), self.learning_gate_count, parameter
            )
        )
        self.learning_gate_count += 1

        if target == _Axis.X:
            self.circuit.add_parametric_RX_gate(index, parameter)
        elif target == _Axis.Y:
            self.circuit.add_parametric_RY_gate(index, parameter)
        elif target == _Axis.Z:
            self.circuit.add_parametric_RZ_gate(index, parameter)
        else:
            raise NotImplementedError

    def add_parametric_RX_gate(self, index: int, parameter: float):
        self._add_parametric_R_gate_inner(index, parameter, _Axis.X)

    def add_parametric_RY_gate(self, index: int, parameter: float):
        self._add_parametric_R_gate_inner(index, parameter, _Axis.Y)

    def add_parametric_RZ_gate(self, index: int, parameter: float):
        self._add_parametric_R_gate_inner(index, parameter, _Axis.Z)

    def _add_input_R_gate_inner(
        self,
        index: int,
        target: _Axis,
        input_func: Callable[[float], float] = lambda x: x,
    ):
        self.parameter_list.append(
            _Parameter(self.circuit.get_parameter_count(), None, None, input_func)
        )

        if target == _Axis.X:
            self.circuit.add_parametric_RX_gate(index, 0.0)
        elif target == _Axis.Y:
            self.circuit.add_parametric_RY_gate(index, 0.0)
        elif target == _Axis.Z:
            self.circuit.add_parametric_RZ_gate(index, 0.0)
        else:
            raise NotImplementedError

    def add_input_RX_gate(
        self,
        index: int,
        input_func: Callable[[float], float] = lambda x: x,
    ):
        self._add_input_R_gate_inner(index, _Axis.X, input_func)

    def add_input_RY_gate(
        self,
        index: int,
        input_func: Callable[[float], float] = lambda x: x,
    ):
        self._add_input_R_gate_inner(index, _Axis.Y, input_func)

    def add_input_RZ_gate(
        self,
        index: int,
        input_func: Callable[[float], float] = lambda x: x,
    ):
        self._add_input_R_gate_inner(index, _Axis.Z, input_func)

    def _add_parametric_input_R_gate_inner(
        self,
        index: int,
        parameter: float,
        target: _Axis,
        input_func: Callable[[float], float] = lambda x: x,
    ):
        self.parameter_list.append(
            _Parameter(
                self.circuit.get_parameter_count(),
                self.learning_gate_count,
                parameter,
                input_func,
            )
        )
        self.learning_gate_count += 1

        if target == _Axis.X:
            self.circuit.add_parametric_RX_gate(index, parameter)
        elif target == _Axis.Y:
            self.circuit.add_parametric_RY_gate(index, parameter)
        elif target == _Axis.Z:
            self.circuit.add_parametric_RZ_gate(index, parameter)
        else:
            raise NotImplementedError

    def add_parametric_input_RX_gate(
        self,
        index: int,
        parameter: float,
        input_func: Callable[[float, float], float] = lambda theta, x: x,
    ):
        self._add_parametric_input_R_gate_inner(index, parameter, _Axis.X, input_func)

    def add_parametric_input_RY_gate(
        self,
        index: int,
        parameter: float,
        input_func: Callable[[float, float], float] = lambda theta, x: x,
    ):
        self._add_parametric_input_R_gate_inner(index, parameter, _Axis.Y, input_func)

    def add_parametric_input_RZ_gate(
        self,
        index: int,
        parameter: float,
        input_func: Callable[[float, float], float] = lambda theta, x: x,
    ):
        self._add_parametric_input_R_gate_inner(index, parameter, _Axis.Z, input_func)

    def update_parameters(self, theta: List[float]):
        for parameter in self.parameter_list:
            if parameter.is_parametric():
                print(parameter.pos, self.circuit.get_parameter_count())
                parameter.set_parameter(theta[parameter.theta_pos])
                if not parameter.is_input():
                    self.circuit.set_parameter(
                        parameter.pos, theta[parameter.theta_pos]
                    )

    def get_parameters(self):
        theta = [p.value for p in self.parameter_list if p.is_parametric()]
        return theta

    def run(self, x: float) -> QuantumState:
        state = QuantumState(self.n_qubit)
        state.set_zero_state()
        for parameter in self.parameter_list:
            if parameter.is_input():
                angle = (
                    parameter.func(parameter.value, x)
                    if parameter.is_parametric()
                    else parameter.func(x)
                )
                self.circuit.set_parameter(parameter.pos, angle)
                parameter.set_parameter(angle)
        self.circuit.update_quantum_state(state)
        return state
