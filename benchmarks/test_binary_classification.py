from binary_classification_skqulacs import binary_classification_skqulacs
from binary_classification_pennylane import binary_classification_pennylane


def test_skqulacs(benchmark):
    n_qubit = 2
    score = benchmark.pedantic(binary_classification_skqulacs, args=[n_qubit], rounds=10)
    assert score > 0.95


def test_pennylane(benchmark):
    n_qubit = 2
    score = benchmark.pedantic(binary_classification_pennylane, rounds=10)
    assert score > 0.95
