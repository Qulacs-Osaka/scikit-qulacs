from skqulacs.circuit import LearningCircuit


def test_equality() -> None:
    circuit1 = LearningCircuit(2)
    circuit1.add_input_RX_gate(0)
    circuit1.add_parametric_input_RY_gate(1, 0.5)
    circuit1.add_parametric_RZ_gate(0, 1.0)

    circuit2 = LearningCircuit(2)
    circuit2.add_input_RX_gate(0)
    circuit2.add_parametric_input_RY_gate(1, 0.5)
    circuit2.add_parametric_RZ_gate(0, 1.0)

    assert circuit1 == circuit2


def test_equality_with_gates_for_different_qubit() -> None:
    circuit1 = LearningCircuit(2)
    circuit1.add_input_RX_gate(0)
    circuit1.add_parametric_input_RY_gate(0, 0.5)
    circuit1.add_parametric_RZ_gate(0, 1.0)

    circuit2 = LearningCircuit(2)
    circuit2.add_input_RX_gate(1)
    circuit2.add_parametric_input_RY_gate(0, 0.5)
    circuit2.add_parametric_RZ_gate(0, 1.0)

    # Limitation: LearningCircuit does not know which qubit each parameter is allocated.
    # For this reason, `circuit1` and `circuit2` are "equal" although they have input RX gates in qubit 0 and 1 respectively.
    assert circuit1 == circuit2


def test_inequality_with_different_parameters() -> None:
    circuit1 = LearningCircuit(2)
    circuit1.add_input_RX_gate(0)
    circuit1.add_parametric_input_RY_gate(0, 10.5)
    circuit1.add_parametric_RZ_gate(0, 1.0)

    circuit2 = LearningCircuit(2)
    circuit2.add_input_RX_gate(0)
    circuit2.add_parametric_input_RY_gate(0, 0.5)
    circuit2.add_parametric_RZ_gate(0, 1.0)

    assert circuit1 != circuit2


def test_parametric_gate() -> None:
    circuit = LearningCircuit(2)
    circuit.add_input_RX_gate(1)
    circuit.add_RX_gate(0, 0.5)
    circuit.add_parametric_RX_gate(0, 0.0)
    circuit.add_parametric_RX_gate(1, 0.0)
    circuit.update_parameters([0.1, 0.2])
    assert [0.1, 0.2] == circuit.get_parameters()


def test_parametric_input_gate() -> None:
    circuit = LearningCircuit(2)
    circuit.add_parametric_input_RX_gate(1, 0.5, lambda theta, x: theta + x[0])
    circuit.run([1.0])
    assert [1.5] == circuit.get_parameters()


def test_parametric_gates_mixed() -> None:
    circuit = LearningCircuit(2)
    circuit.add_parametric_RX_gate(0, 0.1)
    circuit.add_parametric_input_RX_gate(1, 0.5, lambda theta, x: theta + x[0])
    circuit.add_input_RX_gate(0)
    circuit.run([1.0])
    assert [0.1, 1.5] == circuit.get_parameters()

    circuit.update_parameters([0.2, 1.0])
    circuit.run([1.0])
    assert [0.2, 2.0] == circuit.get_parameters()


def test_no_arg_run() -> None:
    circuit = LearningCircuit(2)
    circuit.run()


def test_share_learning_parameter() -> None:
    circuit = LearningCircuit(2)
    circuit.add_parametric_RX_gate(0, 0.0)  # parameter 0.
    circuit.add_parametric_RY_gate(
        1, 0.0, share_with=0
    )  # Compute RY gate with shared parameter 0.
    circuit.update_parameters([0.1])
    assert [0.1] == circuit.get_parameters()


def test_running_shared_parameter() -> None:
    circuit = LearningCircuit(2)
    shared_parameter = circuit.add_parametric_RX_gate(0, 0.0)
    circuit.add_parametric_RY_gate(1, 0.0, share_with=shared_parameter)
    assert [0.0] == circuit.get_parameters()

    circuit_without_share = LearningCircuit(2)
    circuit_without_share.add_parametric_RX_gate(0, 0.0)
    circuit_without_share.add_parametric_RY_gate(1, 0.0)

    circuit.update_parameters([0.1])
    circuit_without_share.update_parameters([0.1, 0.1])

    state = circuit.run([])
    state_without_share = circuit_without_share.run([])
    for v, w in zip(state.get_vector(), state_without_share.get_vector()):
        assert v == w


def test_share_coef_input_learning_parameter() -> None:
    circuit = LearningCircuit(2)
    circuit.add_parametric_RX_gate(0, 0.0)
    shared_parameter = circuit.add_parametric_RX_gate(0, 0.0)
    circuit.add_parametric_RY_gate(
        1, 0.0, share_with=shared_parameter, share_with_coef=2.0
    )
    circuit.update_parameters([0.1, 0.2])
    # パラメータとしては共有しているため2個だけ返る
    res = circuit.get_parameters()
    assert 2 == len(res)
    state = circuit.run([])

    circuit_without_share = LearningCircuit(2)
    circuit_without_share.add_parametric_RX_gate(0, 0.0)
    circuit_without_share.add_parametric_RX_gate(0, 0.0)
    circuit_without_share.add_parametric_RY_gate(1, 0.0)
    # 共有せずに2個目のパラメータを係数を考慮した値にすると一致する
    circuit_without_share.update_parameters([0.1, 0.2, 0.4])
    state_without_share = circuit_without_share.run([])
    for v, w in zip(state.get_vector(), state_without_share.get_vector()):
        assert v == w
