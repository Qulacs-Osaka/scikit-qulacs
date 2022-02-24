from binary_classification_pennylane import binary_classification_pennylane
from binary_classification_skqulacs import (
    binary_classification_skqulacs,
    create_circuit,
    load_dataset,
)


def test_skqulacs(benchmark):
    x_train, x_test, y_train, y_test = load_dataset()
    circuit = create_circuit(6)
    score = benchmark.pedantic(
        binary_classification_skqulacs,
        args=[circuit, x_train, x_test, y_train, y_test],
        rounds=10,
    )
    assert score > 0.95


def test_pennylane(benchmark):
    score = benchmark.pedantic(binary_classification_pennylane, rounds=5)
    assert score > 0.95
