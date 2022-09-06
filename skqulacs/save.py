from pathlib import Path
from pickle import dump, load
from typing import List, Union


def save(parameter: List[float], path: Union[str, Path]) -> None:
    """Save a learning parameter to a pickel file.

    Args:
        model: Learning parameter to save into pickle file.
        path: File path to save the model.
    """
    with open(path, "wb") as f:
        dump(parameter, f, 5)


def restore(path: Union[str, Path]) -> List[float]:
    """Load a learning parameter from a pickel file.

    When you feed the restored parameter to fresh model, you have to call `fit()` with restored parameter because input/output scaler of the model is not initialized.

    Args:
        path: File path from which a learning parameter is loaded.
    """
    with open(path, "rb") as f:
        return load(f)
