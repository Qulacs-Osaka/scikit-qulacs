import random
from typing import Callable, Optional, Tuple

# import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.random import default_rng
from numpy.typing import NDArray
from sklearn.metrics import mean_squared_error

from skqulacs.circuit import create_multi_qubit_param_rotational_ansatz
from skqulacs.qnn import QNNRegressor
from skqulacs.qnn.solver import Grad_Descent, Solver


def generate_noisy_data(
    x_min: float,
    x_max: float,
    x_shape: Tuple[int, int],
    function: Callable[[NDArray[np.float_]], NDArray[np.float_]],
    seed: Optional[int] = 0,
    mag_noise: float = 0.001,
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    """Generate random input data and its output

    Args:
        x_min: Minimum limit of random x value.
        x_max: Maximum limit of random x value.
        x_shape: Shape of x, (batch, features).
        function: Function which generates output from x. It takes 1D feature vector and returns 1D output vector.
        seed: Seed for random value.
        mag_noise: Noise amplitude to be added to output.
    """
    rng = default_rng(seed)
    x_train = rng.uniform(x_min, x_max, x_shape)
    y_train = np.array([function(x) for x in x_train])
    y_train += mag_noise * rng.random(y_train.shape)
    return x_train, y_train


def sine(x: NDArray[np.float_]) -> NDArray[np.float_]:
    return np.sin(np.pi * x[0])


@pytest.mark.parametrize(
    ("solver"),
    [(Grad_Descent())],
)
def test_noisy_sine(
    solver: Solver,
) -> Tuple[
    NDArray[np.float_], NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]
]:
    x_min = -1.0
    x_max = 1.0
    num_x = 300
    x_train, y_train = generate_noisy_data(x_min, x_max, (num_x, 1), sine)
    n_qubit = 3
    depth = 5
    batch_size = 150
    epochs = 1500
    lr = 0.1
    losses = []
    circuit = create_multi_qubit_param_rotational_ansatz(n_qubit, c_depth=depth)
    qnn = QNNRegressor(circuit, solver, observables_str=["Z 2"])
    for epoch in range(epochs):
        indexes = [idx for idx in range(num_x)]
        for bc in range(num_x // batch_size):
            selected = random.sample(indexes, batch_size)
            for el in selected:
                indexes.remove(el)
            x_batch = x_train[selected]
            y_batch = y_train[selected]
            opt_loss, opt_params = qnn.fit(x_batch, y_batch, lr)
            opt_loss = qnn.cost_func(opt_params, x_train, y_train)
            losses.append(opt_loss)
    x_test, y_test = generate_noisy_data(x_min, x_max, (num_x, 1), sine)
    y_pred = qnn.predict(x_test)
    return x_test, y_test, y_pred, losses


def just_gradients(str_ob) -> Tuple[NDArray[np.float_]]:
    x_min = -1.0
    x_max = 1.0
    num_x = 300
    x_train, y_train = generate_noisy_data(x_min, x_max, (num_x, 1), sine)
    n_qubit = 3
    depth = 5
    circuit = create_multi_qubit_param_rotational_ansatz(n_qubit, c_depth=depth)
    qnn = QNNRegressor(circuit, Grad_Descent(), observables_str=str_ob)
    theta = circuit.get_parameters()
    grads_circuit = qnn._func_grad(theta, x_train)
    return grads_circuit


def main() -> None:
    np.random.seed(0)
    random.seed(0)
    x_test, y_test, y_pred, losses = test_noisy_sine(Grad_Descent())
    loss = mean_squared_error(y_pred, y_test)
    print("loss", loss)
    # This is how you compute the gradients of the circuit (based on the observable you select) without Loss
    # see function _func_grad()
    grads_circuit = just_gradients(["Z 1"])
    print(grads_circuit)
    # plt.plot(x_test, y_test, "o", label="Test")
    # plt.plot(x_test, y_pred, "o", label="Prediction")
    # plt.legend()
    # plt.show()
    # plt.savefig("ytest_vs_ypred_batch.jpg")
    # plt.clf()
    # xs = [i for i in range(len(losses))]
    # plt.plot(xs, losses, "o", label="lossesVSiterations")
    # plt.legend()
    # plt.show()
    # plt.savefig("losses_iterations_batch.jpg")


if __name__ == "__main__":
    main()
