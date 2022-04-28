from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

CostFunc = Callable[[List[float], NDArray[np.float_], NDArray[np.float_]], float]
Jacobian = Callable[
    [List[float], NDArray[np.float_], NDArray[np.float_]], NDArray[np.float_]
]


class Solver(ABC):
    @abstractmethod
    def run(
        self,
        cost_func: CostFunc,
        jac: Jacobian,
        theta: List[float],
        x: NDArray[np.float_],
        y: NDArray[np.float_],
        maxiter: Optional[int],
    ) -> Tuple[float, List[float]]:
        """Run optimizer for given initial parameters and data.

        Args:
            theta: Initial value of parameters to optimize.
            x: Data to use in optimization.
            y: Data to use in optimization.
            maxiter: Maximum iteration count in optimization.

        Returns:
            (loss, theta_opt): Loss and parameters after optimization.
        """
        pass


@dataclass
class NelderMead(Solver):
    def run(
        self,
        cost_func: CostFunc,
        jac: Jacobian,
        theta: List[float],
        x: NDArray[np.float_],
        y: NDArray[np.float_],
        maxiter: Optional[int],
    ) -> Tuple[float, List[float]]:
        result = minimize(
            cost_func,
            theta,
            args=(x, y),
            method="Nelder-Mead",
            options={"maxiter": maxiter},
        )
        loss = result.fun
        theta_opt = result.x
        return loss, theta_opt


@dataclass
class Bfgs(Solver):
    def run(
        self,
        cost_func: CostFunc,
        jac: Jacobian,
        theta: List[float],
        x: NDArray[np.float_],
        y: NDArray[np.float_],
        maxiter: Optional[int],
    ) -> Tuple[float, List[float]]:
        result = minimize(
            cost_func,
            theta,
            args=(x, y),
            method="BFGS",
            jac=jac,
            options={"maxiter": maxiter},
        )
        loss = result.fun
        theta_opt = result.x
        return loss, theta_opt


@dataclass
class Adam(Solver):
    callback: Optional[Callable[[List[float]], None]] = None
    tolerance: float = 1e-4
    n_iter_no_change: Optional[int] = None

    def run(
        self,
        cost_func: CostFunc,
        jac: Jacobian,
        theta: List[float],
        x: NDArray[np.float_],
        y: NDArray[np.float_],
        maxiter: Optional[int],
    ) -> Tuple[float, List[float]]:
        pr_A = 0.02
        pr_Bi = 0.8
        pr_Bt = 0.995
        pr_ips = 1e-6
        # Above is hyper parameters.
        Bix = 0.0
        Btx = 0.0

        moment = np.zeros(len(theta))
        vel = 0
        theta_now = theta
        maxiter *= len(x)
        prev_cost = cost_func(theta_now, x, y)

        no_change = 0
        for iter in range(0, maxiter, 5):
            grad = jac(
                theta_now,
                x[iter % len(x) : iter % len(x) + 5],
                y[iter % len(y) : iter % len(y) + 5],
            )
            moment = moment * pr_Bi + (1 - pr_Bi) * grad
            vel = vel * pr_Bt + (1 - pr_Bt) * np.dot(grad, grad)
            Bix = Bix * pr_Bi + (1 - pr_Bi)
            Btx = Btx * pr_Bt + (1 - pr_Bt)
            theta_now -= pr_A / (((vel / Btx) ** 0.5) + pr_ips) * (moment / Bix)
            if (self.n_iter_no_change is not None) and (iter % len(x) < 5):
                if self.callback is not None:
                    self.callback(theta_now)
                now_cost = cost_func(theta_now, x, y)
                if prev_cost - self.tolerance < now_cost:
                    no_change = no_change + 1
                    if no_change >= self.n_iter_no_change:
                        break
                else:
                    no_change = 0
                prev_cost = now_cost

        loss = cost_func(theta_now, x, y)
        theta_opt = theta_now
        return loss, theta_opt
