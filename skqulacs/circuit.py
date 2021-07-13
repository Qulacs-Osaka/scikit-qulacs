from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional, Union
from qulacs import QuantumState, ParametricQuantumCircuit


class _Axis(Enum):
    """Specifying axis. Used in inner private method in LearningCircuit."""
    X = auto()
    Y = auto()
    Z = auto()


@dataclass
class _Parameter:
    """Manage a parameter of ParametricQuantumCircuit.

    Args:
        pos: Index at LearningCircuit._circuit.
        theta_pos: Index at array of theta which are learning parameters.
        value: Current pos-th parameter of LearningCircuit._circuit.
        func: Transforming function for input gate.
    """

    pos: int
    theta_pos: Optional[int]
    value: Optional[float]
    func: Union[
        Callable[[float], float], Callable[[float, float], float], None
    ] = field(default=None)

    def is_learning_parameter(self) -> bool:
        """Return if this parameter is for learning."""
        return self.value is not None and self.theta_pos is not None

    def is_input(self) -> bool:
        return self.func is not None

    def calculate_angle(self, x: float) -> float:
        """Calculate angle for rotation gate from input data x.

        Args:
            x: Input data encoded by self.func.
        """
        if self.is_learning_parameter():
            return self.func(x)
        return self.func(self.value, x)


class LearningCircuit:
    """Construct and run quantum circuit for QNN.

    Args:
        n_qubit: The number of qubits in the circuit.

    Examples:
        >>> from skqulacs.circuit import LearningCircuit
        >>> from skqulacs.qnn.regressor import QNNRegressor
        >>> n_qubit = 2
        >>> circuit = LearningCircuit(n_qubit)
        >>> circuit.add_parametric_RX_gate(0, 0.5)
        >>> circuit.add_input_RZ_gate(1, np.arcsin)
        >>> model = QNNRegressor(n_qubit, circuit)
        >>> _, theta = model.fit(x_train, y_train, maxiter=1000)
        >>> x_list = np.arange(x_min, x_max, 0.02)
        >>> y_pred = qnn.predict(theta, x_list)
    """

    def __init__(
        self,
        n_qubit: int,
    ) -> None:
        self.n_qubit = n_qubit
        self._circuit = ParametricQuantumCircuit(n_qubit)
        self._parameter_list: List[_Parameter] = []
        self._learning_gate_count = 0

    def add_gate(self, gate):
        """Add arbitrary gate.

        Args:
            gate: Gate to add.
        """
        self._circuit.add_gate(gate)

    def add_X_gate(self, index: int):
        """
        Args:
            index: Index of qubit to add X gate.
        """
        self._circuit.add_X_gate(index)

    def add_Y_gate(self, index: int):
        """
        Args:
            index: Index of qubit to add Y gate.
        """
        self._circuit.add_Y_gate(index)

    def add_Z_gate(self, index: int):
        """
        Args:
            index: Index of qubit to add Z gate.
        """
        self._circuit.add_Z_gate(index)

    def add_RX_gate(self, index: int, angle: float):
        """
        Args:
            index: Index of qubit to add RX gate.
            angle: Rotation angle.
        """
        self._add_R_gate_inner(index, angle, _Axis.X)

    def add_RY_gate(self, index: int, parameter: float):
        """
        Args:
            index: Index of qubit to add RY gate.
            angle: Rotation angle.
        """
        self._add_R_gate_inner(index, parameter, _Axis.Y)

    def add_RZ_gate(self, index: int, parameter: float):
        """
        Args:
            index: Index of qubit to add RZ gate.
            angle: Rotation angle.
        """
        self._add_R_gate_inner(index, parameter, _Axis.Z)

    def add_input_RX_gate(
        self,
        index: int,
        input_func: Callable[[float], float] = lambda x: x,
    ):
        """
        Args:
            index: Index of qubit to add RX gate.
            input_func: Function transforming index value.
        """
        self._add_input_R_gate_inner(index, _Axis.X, input_func)

    def add_input_RY_gate(
        self,
        index: int,
        input_func: Callable[[float], float] = lambda x: x,
    ):
        """
        Args:
            index: Index of qubit to add RY gate.
            input_func: Function transforming index value.
        """
        self._add_input_R_gate_inner(index, _Axis.Y, input_func)

    def add_input_RZ_gate(
        self,
        index: int,
        input_func: Callable[[float], float] = lambda x: x,
    ):
        """
        Args:
            index: Index of qubit to add RZ gate.
            input_func: Function transforming index value.
        """
        self._add_input_R_gate_inner(index, _Axis.Z, input_func)

    def add_parametric_RX_gate(self, index: int, parameter: float):
        """
        Args:
            index: Index of qubit to add RX gate.
            parameter: Initial parameter of this gate.
        """
        self._add_parametric_R_gate_inner(index, parameter, _Axis.X)

    def add_parametric_RY_gate(self, index: int, parameter: float):
        """
        Args:
            index: Index of qubit to add RY gate.
            parameter: Initial parameter of this gate.
        """
        self._add_parametric_R_gate_inner(index, parameter, _Axis.Y)

    def add_parametric_RZ_gate(self, index: int, parameter: float):
        """
        Args:
            index: Index of qubit to add RZ gate.
            parameter: Initial parameter of this gate.
        """
        self._add_parametric_R_gate_inner(index, parameter, _Axis.Z)

    def add_parametric_input_RX_gate(
        self,
        index: int,
        parameter: float,
        input_func: Callable[[float, float], float] = lambda theta, x: x,
    ):
        """
        Args:
            index: Index of qubit to add RX gate.
            parameter: Initial parameter of this gate.
            input_func: Function transforming this gate's parameter and index value.
        """
        self._add_parametric_input_R_gate_inner(index, parameter, _Axis.X, input_func)

    def add_parametric_input_RY_gate(
        self,
        index: int,
        parameter: float,
        input_func: Callable[[float, float], float] = lambda theta, x: x,
    ):
        """
        Args:
            index: Index of qubit to add RY gate.
            parameter: Initial parameter of this gate.
            input_func: Function transforming this gate's parameter and index value.
        """
        self._add_parametric_input_R_gate_inner(index, parameter, _Axis.Y, input_func)

    def add_parametric_input_RZ_gate(
        self,
        index: int,
        parameter: float,
        input_func: Callable[[float, float], float] = lambda theta, x: x,
    ):
        """
        Args:
            index: Index of qubit to add RZ gate.
            parameter: Initial parameter of this gate.
            input_func: Function transforming this gate's parameter and index value.
        """
        self._add_parametric_input_R_gate_inner(index, parameter, _Axis.Z, input_func)

    def update_parameters(self, theta: List[float]):
        """Update learning parameter of the circuit.

        Args:
            theta: New learning parameter.
        """
        for parameter in self._parameter_list:
            if parameter.is_learning_parameter():
                parameter.value = theta[parameter.theta_pos]
                if not parameter.is_input():
                    # For non-input parameter, just update.
                    self._circuit.set_parameter(
                        parameter.pos, theta[parameter.theta_pos]
                    )

    def get_parameters(self):
        theta_list = [
            p.value for p in self._parameter_list if p.is_learning_parameter()
        ]
        return theta_list

    def run(self, x: float) -> QuantumState:
        state = QuantumState(self.n_qubit)
        state.set_zero_state()
        for parameter in self._parameter_list:
            if parameter.is_input():
                # Input parameter is updated here, not update_parameters(),
                # because input parameter is determined with input data.
                angle = parameter.calculate_angle(x)
                parameter.value = angle
                self._circuit.set_parameter(parameter.pos, angle)
        self._circuit.update_quantum_state(state)
        return state

    def _add_R_gate_inner(
        self,
        index: int,
        angle: Optional[float],
        target: _Axis,
    ):
        if target == _Axis.X:
            self._circuit.add_RX_gate(index, angle)
        elif target == _Axis.Y:
            self._circuit.add_parametric_RY_gate(index, angle)
        elif target == _Axis.Z:
            self._circuit.add_parametric_RZ_gate(index, angle)
        else:
            raise NotImplementedError

    def _add_parametric_R_gate_inner(
        self,
        index: int,
        parameter: float,
        target: _Axis,
    ):
        self._parameter_list.append(
            _Parameter(
                self._circuit.get_parameter_count(),
                self._learning_gate_count,
                parameter,
            )
        )
        self._learning_gate_count += 1

        if target == _Axis.X:
            self._circuit.add_parametric_RX_gate(index, parameter)
        elif target == _Axis.Y:
            self._circuit.add_parametric_RY_gate(index, parameter)
        elif target == _Axis.Z:
            self._circuit.add_parametric_RZ_gate(index, parameter)
        else:
            raise NotImplementedError

    def _add_input_R_gate_inner(
        self,
        index: int,
        target: _Axis,
        input_func: Callable[[float], float] = lambda x: x,
    ):
        self._parameter_list.append(
            _Parameter(self._circuit.get_parameter_count(), None, None, input_func)
        )

        # Input gate is implemented with parametric gate because this gate should be
        # updated with input data in every iteration.
        if target == _Axis.X:
            self._circuit.add_parametric_RX_gate(index, 0.0)
        elif target == _Axis.Y:
            self._circuit.add_parametric_RY_gate(index, 0.0)
        elif target == _Axis.Z:
            self._circuit.add_parametric_RZ_gate(index, 0.0)
        else:
            raise NotImplementedError

    def _add_parametric_input_R_gate_inner(
        self,
        index: int,
        parameter: float,
        target: _Axis,
        input_func: Callable[[float], float] = lambda x: x,
    ):
        self._parameter_list.append(
            _Parameter(
                self._circuit.get_parameter_count(),
                self._learning_gate_count,
                parameter,
                input_func,
            )
        )
        self._learning_gate_count += 1

        if target == _Axis.X:
            self._circuit.add_parametric_RX_gate(index, parameter)
        elif target == _Axis.Y:
            self._circuit.add_parametric_RY_gate(index, parameter)
        elif target == _Axis.Z:
            self._circuit.add_parametric_RZ_gate(index, parameter)
        else:
            raise NotImplementedError
