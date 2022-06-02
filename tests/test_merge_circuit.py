import pytest
from skqulacs.circuit import LearningCircuit
from skqulacs.circuit import _LearningParameter

def test_merge_circuit_with_shared_parameter():
    circuit1 = LearningCircuit(2)
    circuit1.add_parametric_RX_gate(0, 0.0)
    circuit1.add_parametric_RY_gate(1, 0.0, share_with=0)
    circuit1.add_parametric_RZ_gate(0, 0.0)

    circuit2 = LearningCircuit(2)
    circuit2.add_parametric_RZ_gate(0, 0.0)
    circuit2.add_parametric_RY_gate(1, 0.0)
    circuit2.add_parametric_RZ_gate(0, 0.0, shared_with=1)
    circuit2.add_parametric_RX_gate(0, 0.0, shared_with=1)

    circuit1.merge_circuit(circuit2)
    assert circuit1._learning_parameter_list[0] == _LearningParameter([0, 1])
    assert circuit1._learning_parameter_list[3] == _LearningParameter([5, 6, 7])

def test_merge_circuit_with_shared_input_parameter():
    # 上のテストを input parameter を含む回路でやる
    ...

def test_invalid_merge_of_circuits_with_different_qubits():
    circuit1 = LearningCircuit(2)
    circuit2 = LearningCircuit(3)
    # c.f. https://docs.pytest.org/en/7.1.x/how-to/assert.html?highlight=exception#assertions-about-expected-exceptions
    with pytest.raises(RuntimeError):
        circuit1.merge_circuit(circuit2)

if __name__ == "__main__":
    test_merge_circuit_with_shared_parameter()
    test_merge_circuit_with_shared_input_parameter()
    test_invalid_merge_of_circuits_with_different_qubits()
