from skqulacs.circuit import LearningCircuit


def test_parametric_gate():
    circuit = LearningCircuit(2)
    circuit.add_input_RX_gate(1)
    circuit.add_RX_gate(0, 0.5)
    circuit.add_parametric_RX_gate(0, 0.0)
    circuit.add_parametric_RX_gate(1, 0.0)
    circuit.update_parameters([0.1, 0.2])
    assert [0.1, 0.2] == circuit.get_parameters()


def test_parametric_input_gate():
    circuit = LearningCircuit(2)
    circuit.add_parametric_input_RX_gate(1, 0.5, lambda theta, x: theta + x[0])
    circuit.run([1.0])
    assert [1.5] == circuit.get_parameters()


def test_parametric_gates_mixed():
    circuit = LearningCircuit(2)
    circuit.add_parametric_RX_gate(0, 0.1)
    circuit.add_parametric_input_RX_gate(1, 0.5, lambda theta, x: theta + x[0])
    circuit.add_input_RX_gate(0)
    circuit.run([1.0])
    assert [0.1, 1.5] == circuit.get_parameters()

    circuit.update_parameters([0.2, 1.0])
    circuit.run([1.0])
    assert [0.2, 2.0] == circuit.get_parameters()

def test_no_arg_run():
    circuit = LearningCircuit(2)
    state = circuit.run()
    print(state.get_vector())
