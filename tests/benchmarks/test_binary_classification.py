import pytest
from binary_classification_skqulacs import binary_classification


def test_binary_classification(benchmark):
    n_qubit = 2
    score = benchmark.pedantic(binary_classification, args=[n_qubit], rounds=10)
    assert score > 0.95
