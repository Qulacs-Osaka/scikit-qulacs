from pathlib import Path

from sklearn.metrics import mean_squared_error
from test_qnn_regressor import generate_noisy_data, sine

from skqulacs.circuit import create_qcl_ansatz
from skqulacs.qnn import QNNRegressor
from skqulacs.qnn.solver import Bfgs
from skqulacs.save import restore, save


# `tmp_path` is a temporary file fixture provided by pytest.
def test_save_and_load_parameter_for_regressor(tmp_path: Path) -> None:
    x_min = -1.0
    x_max = 1.0
    num_x = 50
    x_train, y_train = generate_noisy_data(x_min, x_max, (num_x, 1), sine)

    circuit = create_qcl_ansatz(3, 3, 0.5, 0)
    qnn = QNNRegressor(circuit, Bfgs())
    qnn.fit(x_train, y_train, 20)

    # Save parameter.
    parameter_path = tmp_path / "parameter.pickle"
    parameter = qnn.circuit.get_parameters()
    save(parameter, parameter_path)
    del circuit
    del qnn

    # Restore parameter.
    parameter_loaded = restore(parameter_path)
    assert parameter_loaded == parameter

    # Rebuild model with restored parameter.
    x_test, y_test = generate_noisy_data(x_min, x_max, (num_x, 1), sine)
    circuit = create_qcl_ansatz(3, 3, 0.5, 0)
    qnn = QNNRegressor(circuit, Bfgs())
    qnn.circuit.update_parameters(parameter_loaded)
    qnn.fit(x_train, y_train, 1)

    y_pred = qnn.predict(x_test)
    loss = mean_squared_error(y_pred, y_test)
    assert loss < 0.03
