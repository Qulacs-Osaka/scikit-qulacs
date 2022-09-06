from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional, Union

import numpy as np
from numpy.typing import NDArray
from qulacs import ParametricQuantumCircuit, QuantumState


class _Axis(Enum):
    """Specifying axis. Used in inner private method in LearningCircuit."""

    X = auto()
    Y = auto()
    Z = auto()


# Depends on x
InputFunc = Callable[[NDArray[np.float_]], float]

# Depends on theta, x
InputFuncWithParam = Callable[[float, NDArray[np.float_]], float]


@dataclass
class _PositionDetail:
    """Manage a parameter of `ParametricQuantumCircuit.positions_in_circuit`.
    This class manages indexe and coefficients (optional) of gate.
    Args:
        gate_pos: Indices of a parameter in LearningCircuit._circuit.
        coef: Coefficient of a parameter in LearningCircuit._circuit. It's a optional.
    """

    gate_pos: int
    coef: Optional[float]


@dataclass
class _LearningParameter:
    """Manage a parameter of `ParametricQuantumCircuit`.
    This class manages index and value of parameter.
    There is two member variables to note: `positions_in_circuit` and `parameter_id`.
    `positions_in_circuit` is indices of parameters in `ParametricQuantumCircuit` held by `LearningCircuit`.
    If you change the parameter value of the `_LearningParameter` instance, all of the parameters
    specified in `positions_in_circuit` are also updated with that value.
    And `parameter_id` is an index of a whole set of learning parameters.
    This is used by method of `LearningCircuit` which has "parametric" in its name.

    Args:
        positions_in_circuit: Indices and coefficient of a parameter in LearningCircuit._circuit.
        parameter_id: Index at array of learning parameter(theta).
        value: Current `parameter_id`-th parameter of LearningCircuit._circuit.
        is_input: Whethter this parameter is used with a input parameter.
    """

    positions_in_circuit: List[_PositionDetail]
    parameter_id: int
    value: float
    is_input: bool = field(default=False)

    def __init__(self, parameter_id, value, is_input=False) -> None:
        self.positions_in_circuit = []
        self.parameter_id = parameter_id
        self.value = value
        self.is_input = is_input

    def append_position(self, position: int, coef: Optional[float]) -> None:
        self.positions_in_circuit.append(_PositionDetail(position, coef))


@dataclass
class _InputParameter:
    """Manage transformation of an input.
    `func` transforms the given input and the outcome is stored at `pos`-th parameter in `LearningCircuit._circuit`.
    If the `func` needs a learning parameter, supply `companion_parameter_id` with the learning parameter's `parameter_id`.
    """

    pos: int
    func: Union[InputFunc, InputFuncWithParam] = field(compare=False)
    companion_parameter_id: Optional[int]


@dataclass(eq=False)
class LearningCircuit:
    """Construct and run quantum circuit for QNN.

    ## About parameters

    This class manages parameters of underlying `ParametricQuantumCircuit`.
    A parameter has either type of features: learning and input.

    Learning parameter represents a parameter to be optimized.
    This is updated by `LearningCircuit.update_parameter()`.

    Input parameter represents a placeholder of circuit input.
    This is updated in a execution of `LearningCircuit.run()` while applying `func` of the parameter.

    And there is a parameter being both learning and input one.
    This parameter transforms its input by applying the parameter's `func` with its learning parameter.

    ## Execution flow

    1. Set up gates by `LearningCircuit.add_*_gate()`.
    2. For each execution, at first, feed input parameter with the value computed from input data `x`.
    3. Apply |0> state to the circuit.
    4. Compute optimized learning parameters in a certain way.
    5. Update the learning parameters in the circuit with the optimized ones by `LearningCircuit.update_parameters()`.

    Args:
        n_qubit: The number of qubits in the circuit.

    Examples:
        >>> from skqulacs.circuit import LearningCircuit
        >>> from skqulacs.qnn.regressor import QNNRegressor
        >>> n_qubit = 2
        >>> circuit = LearningCircuit(n_qubit)
        >>> theta = circuit.add_parametric_RX_gate(0, 0.5)
        >>> circuit.add_parametric_RY_gate(1, 0.1, share_with=theta)
        >>> circuit.add_input_RZ_gate(1, np.arcsin)
        >>> model = QNNRegressor(circuit)
        >>> _, theta = model.fit(x_train, y_train, maxiter=1000)
        >>> x_list = np.arange(x_min, x_max, 0.02)
        >>> y_pred = qnn.predict(theta, x_list)
    """

    n_qubit: int
    # ParametricQuantumCircuit does not have a function to compare by value, so exclude from comparison of LearningCircuit for now.
    _circuit: ParametricQuantumCircuit = field(init=False, compare=False)
    _learning_parameter_list: List[_LearningParameter] = field(
        init=False, default_factory=list
    )
    _input_parameter_list: List[_InputParameter] = field(
        init=False, default_factory=list
    )

    def __post_init__(self) -> None:
        self._circuit = ParametricQuantumCircuit(self.n_qubit)

    def update_parameters(self, theta: List[float]) -> None:
        """Update learning parameter of the circuit with given `theta`.

        Args:
            theta: New learning parameters.
        """
        for parameter in self._learning_parameter_list:
            parameter_value = theta[parameter.parameter_id]
            parameter.value = parameter_value
            for pos in parameter.positions_in_circuit:
                self._circuit.set_parameter(
                    pos.gate_pos,
                    parameter_value * (pos.coef or 1.0),
                )

    def get_parameters(self) -> List[float]:
        """Get a list of learning parameters' values."""
        theta_list = [p.value for p in self._learning_parameter_list]
        return theta_list

    def _set_input(self, x: NDArray[np.float_]) -> None:
        for parameter in self._input_parameter_list:
            # Input parameter is updated here, not update_parameters(),
            # because input parameter is determined with the input data `x`.
            if parameter.companion_parameter_id is None:
                # If `companion_parameter_id` is `None`, `func` does not need a learning parameter.
                angle = parameter.func(x)
            else:
                theta = self._learning_parameter_list[parameter.companion_parameter_id]
                angle = parameter.func(theta.value, x)
                theta.value = angle
            self._circuit.set_parameter(parameter.pos, angle)

    def run(self, x: List[float] = list()) -> QuantumState:
        """Determine parameters for input gate based on `x` and apply the circuit to |0> state.

        Arguments:
            x: Input data whose shape is (n_features,).

        Returns:
            Quantum state applied the circuit.
        """
        state = QuantumState(self.n_qubit)
        state.set_zero_state()
        self._set_input(x)
        self._circuit.update_quantum_state(state)
        return state

    def run_x_no_change(self) -> QuantumState:
        """
        Run the circuit while x is not changed from the previous run.
        (can change parameters)
        """
        state = QuantumState(self.n_qubit)
        state.set_zero_state()
        self._circuit.update_quantum_state(state)
        return state

    def backprop(self, x: List[float], obs) -> List[float]:
        """
        backprop(self, x: List[float], obs)->List[Float]

        xは入力の状態で、yは出力値の微分値
        帰ってくるのは、それぞれのパラメータに関する微分値
        例えば、出力が[0,2]
        だったらパラメータの1項目は期待する出力に関係しない、2項目をa上げると回路の出力は2a上がる?

        ->
        c++のParametricQuantumCircuitクラスを呼び出す
        backprop(GeneralQuantumOperator* obs)

        ->うまくやってbackpropする。
        現実だと不可能な演算も含むが、気にしない
        """
        self._set_input(x)
        ret = self._circuit.backprop(obs)
        ans = [0.0] * len(self._learning_parameter_list)
        for parameter in self._learning_parameter_list:
            if not parameter.is_input:
                for pos in parameter.positions_in_circuit:
                    ans[parameter.parameter_id] += ret[pos.gate_pos] * (pos.coef or 1.0)

        return ans

    def backprop_inner_product(self, x: List[float], state) -> List[float]:
        """
        backprop(self, x: List[float],  state)->List[Float]

        inner_productでbackpropします。
        """
        self._set_input(x)
        ret = self._circuit.backprop_inner_product(state)
        ans = [0.0] * len(self._learning_parameter_list)
        for parameter in self._learning_parameter_list:
            if not parameter.is_input:
                for pos in parameter.positions_in_circuit:
                    ans[parameter.parameter_id] += ret[pos.gate_pos] * (pos.coef or 1.0)

        return ans

    def _new_parameter_position(self) -> int:
        """Return a position of a new parameter to be registered to `ParametricQuantumCircuit`.
        This function does not actually register a new parameter.
        """
        return self._circuit.get_parameter_count()

    def add_gate(self, gate) -> None:
        """Add arbitrary gate.

        Args:
            gate: Gate to add.
        """
        self._circuit.add_gate(gate)

    def add_X_gate(self, index: int) -> None:
        """
        Args:
            index: Index of qubit to add X gate.
        """
        self._circuit.add_X_gate(index)

    def add_Y_gate(self, index: int) -> None:
        """
        Args:
            index: Index of qubit to add Y gate.
        """
        self._circuit.add_Y_gate(index)

    def add_Z_gate(self, index: int) -> None:
        """
        Args:
            index: Index of qubit to add Z gate.
        """
        self._circuit.add_Z_gate(index)

    def add_RX_gate(self, index: int, angle: float) -> None:
        """
        Args:
            index: Index of qubit to add RX gate.
            angle: Rotation angle.
        """
        self._add_R_gate_inner(index, angle, _Axis.X)

    def add_RY_gate(self, index: int, parameter: float) -> None:
        """
        Args:
            index: Index of qubit to add RY gate.
            angle: Rotation angle.
        """
        self._add_R_gate_inner(index, parameter, _Axis.Y)

    def add_RZ_gate(self, index: int, parameter: float) -> None:
        """
        Args:
            index: Index of qubit to add RZ gate.
            angle: Rotation angle.
        """
        self._add_R_gate_inner(index, parameter, _Axis.Z)

    def add_CNOT_gate(self, control_index: int, target_index: int) -> None:
        """
        Args:
            control_index: Index of control qubit.
            target_index: Index of target qubit.
        """
        self._circuit.add_CNOT_gate(control_index, target_index)

    def add_H_gate(self, index: int) -> None:
        """
        Args:
            index: Index of qubit to put H gate.
        """
        self._circuit.add_H_gate(index)

    def add_input_RX_gate(
        self,
        index: int,
        input_func: InputFunc = lambda x: x[0],
    ) -> None:
        """
        Args:
            index: Index of qubit to add RX gate.
            input_func: Function transforming input value.
        """
        self._add_input_R_gate_inner(index, _Axis.X, input_func)

    def add_input_RY_gate(
        self,
        index: int,
        input_func: InputFunc = lambda x: x[0],
    ) -> None:
        """
        Args:
            index: Index of qubit to add RY gate.
            input_func: Function transforming input value.
        """
        self._add_input_R_gate_inner(index, _Axis.Y, input_func)

    def add_input_RZ_gate(
        self,
        index: int,
        input_func: InputFunc = lambda x: x[0],
    ) -> None:
        """
        Args:
            index: Index of qubit to add RZ gate.
            input_func: Function transforming input value.
        """
        self._add_input_R_gate_inner(index, _Axis.Z, input_func)

    def add_parametric_RX_gate(
        self,
        index: int,
        parameter: float,
        share_with: Optional[int] = None,
        share_with_coef: Optional[float] = None,
    ) -> int:
        """
        Args:
            index: Index of qubit to add RX gate.
            parameter: Initial parameter of this gate.
            share_with: parameter_id to share the parameter in `ParametricQuantumCircuit`.
            share_with_coef: Coefficients for shared parameters which is `share_with`. if 'share_with' is none, share_with_coef is skiped.

        Returns:
            parameter_id which is added or updated.
        """
        return self._add_parametric_R_gate_inner(
            index, parameter, _Axis.X, share_with, share_with_coef
        )

    def add_parametric_RY_gate(
        self,
        index: int,
        parameter: float,
        share_with: Optional[int] = None,
        share_with_coef: Optional[float] = None,
    ) -> int:
        """
        Args:
            index: Index of qubit to add RY gate.
            parameter: Initial parameter of this gate.
            share_with: parameter_id to share the parameter in `ParametricQuantumCircuit`.
            share_with_coef: Coefficients for shared parameters which is `share_with`.

        Returns:
            parameter_id which is added or updated.
        """
        return self._add_parametric_R_gate_inner(
            index, parameter, _Axis.Y, share_with, share_with_coef
        )

    def add_parametric_RZ_gate(
        self,
        index: int,
        parameter: float,
        share_with: Optional[int] = None,
        share_with_coef: Optional[float] = None,
    ) -> int:
        """
        Args:
            index: Index of qubit to add RZ gate.
            parameter: Initial parameter of this gate.
            share_with: parameter_id to share the parameter in `ParametricQuantumCircuit`.
            share_with_coef: Coefficients for shared parameters which is `share_with`. if 'share_with' is none, share_with_coef is skiped.

        Returns:
            parameter_id which is added or updated.
        """
        return self._add_parametric_R_gate_inner(
            index, parameter, _Axis.Z, share_with, share_with_coef
        )

    def add_parametric_input_RX_gate(
        self,
        index: int,
        parameter: float,
        input_func: InputFuncWithParam = lambda theta, x: x[0],
    ) -> None:
        """
        Args:
            index: Index of qubit to add RX gate.
            parameter: Initial parameter of this gate.
            input_func: Function transforming this gate's parameter and input value.
        """
        self._add_parametric_input_R_gate_inner(index, parameter, _Axis.X, input_func)

    def add_parametric_input_RY_gate(
        self,
        index: int,
        parameter: float,
        input_func: InputFuncWithParam = lambda theta, x: x[0],
    ) -> None:
        """
        Args:
            index: Index of qubit to add RY gate.
            parameter: Initial parameter of this gate.
            input_func: Function transforming this gate's parameter and input value.
        """
        self._add_parametric_input_R_gate_inner(index, parameter, _Axis.Y, input_func)

    def add_parametric_input_RZ_gate(
        self,
        index: int,
        parameter: float,
        input_func: InputFuncWithParam = lambda theta, x: x[0],
    ) -> None:
        """
        Args:
            index: Index of qubit to add RZ gate.
            parameter: Initial parameter of this gate.
            input_func: Function transforming this gate's parameter and input value.
        """
        self._add_parametric_input_R_gate_inner(index, parameter, _Axis.Z, input_func)

    def _add_R_gate_inner(
        self,
        index: int,
        angle: Optional[float],
        target: _Axis,
    ) -> None:
        if target == _Axis.X:
            self._circuit.add_RX_gate(index, angle)
        elif target == _Axis.Y:
            self._circuit.add_RY_gate(index, angle)
        elif target == _Axis.Z:
            self._circuit.add_RZ_gate(index, angle)
        else:
            raise NotImplementedError

    def _add_parametric_R_gate_inner(
        self,
        index: int,
        parameter: float,
        target: _Axis,
        share_with: Optional[int],
        share_with_coef: Optional[float],
    ) -> int:
        new_gate_pos = self._new_parameter_position()

        if share_with is None:
            parameter_id = len(self._learning_parameter_list)
            learning_parameter = _LearningParameter(
                parameter_id,
                parameter,
            )
            learning_parameter.append_position(new_gate_pos, None)
            self._learning_parameter_list.append(learning_parameter)
        else:
            parameter_id = share_with
            sharing_parameter = self._learning_parameter_list[parameter_id]
            sharing_parameter.append_position(new_gate_pos, share_with_coef)

        if target == _Axis.X:
            self._circuit.add_parametric_RX_gate(index, parameter)
        elif target == _Axis.Y:
            self._circuit.add_parametric_RY_gate(index, parameter)
        elif target == _Axis.Z:
            self._circuit.add_parametric_RZ_gate(index, parameter)
        else:
            raise NotImplementedError

        return parameter_id

    def _add_input_R_gate_inner(
        self,
        index: int,
        target: _Axis,
        input_func: InputFunc,
    ) -> None:
        self._input_parameter_list.append(
            _InputParameter(self._new_parameter_position(), input_func, None)
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
        input_func: InputFuncWithParam,
    ) -> None:
        pos = self._circuit.get_parameter_count()

        learning_parameter = _LearningParameter(
            len(self._learning_parameter_list), parameter, True
        )
        learning_parameter.append_position(pos, None)
        self._learning_parameter_list.append(learning_parameter)

        self._input_parameter_list.append(
            _InputParameter(pos, input_func, learning_parameter.parameter_id)
        )

        if target == _Axis.X:
            self._circuit.add_parametric_RX_gate(index, parameter)
        elif target == _Axis.Y:
            self._circuit.add_parametric_RY_gate(index, parameter)
        elif target == _Axis.Z:
            self._circuit.add_parametric_RZ_gate(index, parameter)
        else:
            raise NotImplementedError

    def add_parametric_multi_Pauli_rotation_gate(
        self, target: List[int], pauli_id: List[int], initial_angle: float
    ):
        self._circuit.add_parametric_multi_Pauli_rotation_gate(
            target, pauli_id, initial_angle
        )

    def get_circuit_info(self):
        return self._circuit

    def get_circuit_depth(self):
        return self._circuit.calculate_depth()
