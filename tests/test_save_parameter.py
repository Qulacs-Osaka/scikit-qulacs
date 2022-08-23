from pathlib import Path

import pandas as pd
from sklearn import datasets
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from test_qnn_regressor import generate_noisy_data, sine

from skqulacs.circuit import create_qcl_ansatz
from skqulacs.qnn import QNNClassifier, QNNRegressor
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
    parameter = qnn.circuit.get_parameters()
    parameter_path = tmp_path / "parameter.pickle"
    save(parameter, parameter_path)

    # Restore parameter.
    parameter_loaded = restore(parameter_path)
    assert parameter_loaded == parameter
    del qnn
    del circuit
    del parameter

    # Rebuild model with restored parameter.
    x_test, y_test = generate_noisy_data(x_min, x_max, (num_x, 1), sine)
    circuit = create_qcl_ansatz(3, 3, 0.5, 0)
    qnn = QNNRegressor(circuit, Bfgs())
    qnn.circuit.update_parameters(parameter_loaded)
    qnn.fit(x_train, y_train, 0)

    y_pred = qnn.predict(x_test)
    loss = mean_squared_error(y_pred, y_test)
    assert loss < 0.03


def test_save_and_load_parameter_for_classifier(tmp_path: Path) -> None:
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    x = df.loc[:, ["petal length (cm)", "petal width (cm)"]]
    x_train, x_test, y_train, y_test = train_test_split(
        x, iris.target, test_size=0.25, random_state=0
    )
    x_train = x_train.to_numpy()
    x_test = x_test.to_numpy()

    circuit = create_qcl_ansatz(5, 3, 0.5, 0)
    qcl = QNNClassifier(circuit, 3, Bfgs())
    qcl.fit(x_train, y_train, 8)

    parameter_path = tmp_path / "parameter.pickle"
    parameter = qcl.circuit.get_parameters()
    save(parameter, parameter_path)

    parameter_loaded = restore(parameter_path)
    assert parameter_loaded == parameter
    del qcl
    del circuit
    del parameter

    circuit = create_qcl_ansatz(5, 3, 0.5, 0)
    qcl = QNNClassifier(circuit, 3, Bfgs())
    qcl.circuit.update_parameters(parameter_loaded)
    qcl.fit(x_train, y_train, 0)
    y_pred = qcl.predict(x_test)

    assert f1_score(y_test, y_pred, average="weighted") > 0.94
