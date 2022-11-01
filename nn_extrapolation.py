from typing import Optional

import torch
from torch.nn import utils
from torch.optim import SGD

from extrapolation import difference_matrix, regularized_RRE, RRE, levin_transform


def _bound_append(xs, x, k):
    xs = xs + [x]
    return xs[-k:]


class AcceleratedSGD(SGD):
    """SGD implementation with acceleration.

    An SGD implementation that tracks model parameters at each iteration, allowing
    an acceleration algorithm to be run.

    Available methods:
    * RNA - Regularized Nonlinear Acceleration, i.e. RRE with a regularization term defined by a constant lambda
    * RRE - RRE implemented with QR factorization
    * levin:t, levin:u, levin:v - variants of the vector Levin acceleration
    """

    MODES = ["epoch", "epoch_avg", "step"]
    METHODS = ["rna", "rre", "levin:t", "levin:u", "levin:v"]

    def __init__(self, params, lr: float, k: int = 10, lambda_: float = 1e-10, momentum: float = 0,
                 dampening: float = 0, weight_decay: float = 0, nesterov: bool = False,
                 mode: str = "epoch", method: Optional[str] = "RNA", avg_alpha: Optional[int] = None,
                 avg_copy_to_cpu: bool = False):
        """Initialization.

        :param params: model parameters or parameter groups
        :param lr: learning rate
        :param k: number of samples used for acceleration
        :param lambda_: lambda parameter for the acceleration method
        :param momentum: momentum factor, passed to SGD
        :param dampening: dampening for momentum, passed to SGD
        :param weight_decay: weight decay, passed to SGF
        :param nesterov: enables Nesterov momentum, passed to SGD
        :param mode: how samples for acceleration are picked. Available options are step - after each iteration,
            epoch - after each epoch, epoch_avg - average calculated from each epoch
        :param method: acceleration method, or None for no acceleration
        :param avg_alpha: alpha for the exponential moving average, or None to use the arithmetic mean
        :param avg_copy_to_cpu:
        """
        self.mode = mode
        self.avg_alpha = avg_alpha
        self.avg_copy_to_cpu = avg_copy_to_cpu
        assert mode in self.MODES, f"Unknown mode: {mode}, available options: {self.MODES}"
        assert method is None or method.lower() in self.METHODS, \
            f"Unknown method: {method}, available options: {self.METHODS}"
        assert k > 0 or method is None, "Acceleration methods require k > 0"

        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        # add custom parameters
        self.defaults.update(
            lambda_=lambda_,
            k=k,
            method=method,
        )
        for group in self.param_groups:
            for key in ["method", "k", "lambda_"]:
                group.setdefault(key, self.defaults[key])
        self.reset_stored_params()

    def add_param_group(self, param_group: dict):
        super().add_param_group(param_group)
        self._reset_group_params(self.param_groups[-1])

    def reset_stored_params(self):
        """Clear stored parameters."""
        for group in self.param_groups:
            self._reset_group_params(group)

    def store_parameters(self, target_groups=None):
        """Store parameters calculated during acceleration in the model.

        :param target_groups: Parameter groups of the target model. Pass none to save in the model passed in __init__
        """
        if target_groups is None:
            for group in self.param_groups:
                if "accelerated_params" in group:
                    utils.vector_to_parameters(group["accelerated_params"], group["params"])
        else:
            for group, target in zip(self.param_groups, target_groups):
                if "accelerated_params" in group:
                    utils.vector_to_parameters(group["accelerated_params"], target)

    def step(self, closure=None):
        super().step(closure)
        if self.mode == "step":
            self._update_stored_params()
        elif self.mode == "epoch_avg":
            self._update_param_avg()

    def finish_epoch(self):
        """Perform parameter updates required after each epoch."""
        if self.mode == "epoch":
            self._update_stored_params()
        elif self.mode == "epoch_avg":
            self._update_stored_params_from_avg()

    def accelerate(self):
        """Run the acceleration algorithm.

        This method does not modify the model - use store_parameters afterwards.
        """
        for group in self.param_groups:
            if group.get("method") is None:
                continue
            method = group["method"].lower()
            xs = list(group["stored_params"])
            if len(xs) < group["k"]:
                raise ValueError("Not enough stored values to accelerate")
            xs = xs[-group["k"]:]
            if method in ["rna", "rre"]:
                U = difference_matrix(xs)
                X = torch.vstack(xs[1:])
                if method == "rna":
                    group["accelerated_params"] = regularized_RRE(X, U, group["lambda_"])
                elif method == "rre":
                    group["accelerated_params"] = RRE(X, U)
            elif method.startswith("levin:"):
                X = torch.vstack(xs)
                levin_type = method[-1]
                if levin_type == "v":
                    k = group["k"] - 3
                else:
                    k = group["k"] - 2
                group["accelerated_params"] = levin_transform(X, k, levin_type).ravel()

    @staticmethod
    def _reset_group_params(group):
        if group.get("method") is not None:
            group["stored_params"] = []
            group.pop("accelerated_params", None)
            group.pop("stored_params_avg", None)
            group.pop("stored_params_window", None)

    def _update_stored_params(self):
        for group in self.param_groups:
            if group.get("method") is not None:
                x = utils.parameters_to_vector(group["params"]).detach().cpu()
                group["stored_params"] = _bound_append(group["stored_params"], x, group["k"])

    def _update_param_avg(self):
        for group in self.param_groups:
            if group.get("method") is None:
                continue
            x = utils.parameters_to_vector(group["params"]).detach()
            if self.avg_copy_to_cpu:
                x = x.cpu()
            if "stored_params_avg" in group:
                group["stored_params_ctr"] += 1
                if self.avg_alpha is None:
                    c = 1 / group["stored_params_ctr"]
                else:
                    c = self.avg_alpha
                group["stored_params_avg"] = c * x + (1 - c) * group["stored_params_avg"]
            else:
                group["stored_params_avg"] = x
                group["stored_params_ctr"] = 1

    def _update_stored_params_from_avg(self):
        for group in self.param_groups:
            if group.get("method") is not None:
                x = group["stored_params_avg"].cpu()
                group["stored_params"] = _bound_append(group["stored_params"], x, group["k"])
