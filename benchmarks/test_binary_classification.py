from binary_classification_pennylane import (
    binary_classification_pennylane,
    load_iris_pennylane,
)
from binary_classification_skqulacs import (
    binary_classification_skqulacs,
    create_circuit,
    load_iris_skqulacs,
)


def test_skqulacs(benchmark):
    x_train, x_test, y_train, y_test = load_iris_skqulacs()
    circuit = create_circuit(6)
    score = benchmark.pedantic(
        binary_classification_skqulacs,
        args=[circuit, x_train, x_test, y_train, y_test],
        rounds=5,
    )
    assert score > 0.95


def test_pennylane(benchmark):
    x_train, x_val, x_test, y_train, y_val, y_test = load_iris_pennylane()
    score = benchmark.pedantic(
        binary_classification_pennylane,
        args=[x_train, x_val, x_test, y_train, y_val, y_test],
        rounds=5,
    )
    assert score > 0.95
