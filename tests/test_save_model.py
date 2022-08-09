from pathlib import Path

from skqulacs.circuit import create_qcl_ansatz
from skqulacs.qnn import QNNRegressor
from skqulacs.qnn.solver import Bfgs
from skqulacs.save import load, save


# `tmp_path` is a temporary file fixture provided by pytest.
def _test_save_and_load_regressor(tmp_path: Path) -> None:
    n_qubit = 3
    depth = 3
    time_step = 0.5
    circuit = create_qcl_ansatz(n_qubit, depth, time_step, 0)
    qnn = QNNRegressor(circuit, Bfgs())

    save(qnn, tmp_path)

    qnn_loaded: QNNRegressor = load(tmp_path)
    qnn.__eq__(qnn_loaded)
    assert qnn.circuit == qnn_loaded.circuit
