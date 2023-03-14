import numpy as np
import pytest

from skqulacs.dataloader import DataLoader


def test_dataloader_batches_has_correct_size() -> None:
    x = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([1, 2, 3, 4, 5])
    loader = DataLoader(x, y, batch_size=2)
    for x_batch, y_batch in loader:
        assert len(x_batch) <= 2
        assert len(y_batch) <= 2


def test_dataloader_x_and_y_have_different_length() -> None:
    x = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([1, 2, 3, 4])
    with pytest.raises(ValueError):
        DataLoader(x, y)


def test_dataloader_len() -> None:
    x = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([1, 2, 3, 4, 5])
    loader = DataLoader(x, y, batch_size=2)
    assert len(loader) == 3
