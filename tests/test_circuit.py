from skqulacs.circuit import LearningCircuit
import numpy as np


def test_parameter():
    circuit = LearningCircuit(2, 3, 0.0)
    # input かどうかを判別する必要がある
    # input に対しては parameter を渡す必要はない
    # input と parametric で4通りある
    circuit.add_input_RX_gate(1)
    circuit.add_RX_gate(0, 0.5)
    circuit.add_parametric_RX_gate(0, 0.0)
    circuit.add_parametric_RX_gate(1, 0.0)
    circuit.update_noninput_parameter([0.1, 0.2])
    # 0.0 is input_RX_gate's one.
    assert [0.0, 0.1, 0.2] == circuit.get_parameter()


def test_input():
    circuit = LearningCircuit(2, 3, 0.0)
    circuit.add_input_RX_gate(1, lambda x: x * 2.0)
    circuit.run(0.5)
    assert [1.0] == circuit.get_parameter()
