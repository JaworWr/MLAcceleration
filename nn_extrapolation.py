from collections import deque
from typing import Optional

import torch
from torch.nn import utils
from torch.optim import SGD

from extrapolation import difference_matrix, regularized_RRE, RRE


class AcceleratedSGD(SGD):

    def __init__(self, params, lr: float, k: int = 10, lambda_: float = 1e-10, momentum: float = 0,
                 dampening: float = 0, weight_decay: float = 0, nesterov: bool = False,
                 mode: str = "epoch", method: Optional[str] = "RNA"):
        self.mode = mode
        assert method in ["RNA", "RRE", None], "Unknown method: " + method
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

    def _reset_group_params(self, group):
        if group.get("method") is not None:
            group["stored_params"] = deque([], maxlen=group["k"])
            group.pop("accelerated_params", None)
            group.pop("stored_params_avg", None)

    def reset_stored_params(self):
        for group in self.param_groups:
            self._reset_group_params(group)

    def update_stored_params(self):
        for group in self.param_groups:
            if group.get("method") is not None:
                x = utils.parameters_to_vector(group["params"]).cpu()
                group["stored_params"].append(x)

    def update_param_avg(self):
        for group in self.param_groups:
            if group.get("method") is None:
                continue
            x = utils.parameters_to_vector(group["params"]).cpu()
            if "stored_params_avg" in group:
                group["stored_params_ctr"] += 1
                c = 1 / group["stored_params_ctr"]
                group["stored_params_avg"] = c * x + (1 - c) * group["stored_params_avg"]
            else:
                group["stored_params_avg"] = x
                group["stored_params_ctr"] = 1

    def update_stored_params_from_avg(self):
        for group in self.param_groups:
            if group.get("method") is not None:
                x = group["stored_params_avg"]
                group["stored_params"].append(x)

    def store_parameters(self, target_groups=None):
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
            self.update_stored_params()
        elif self.mode == "epoch_avg":
            self.update_param_avg()

    def finish_epoch(self):
        if self.mode == "epoch":
            self.update_stored_params()
        elif self.mode == "epoch_avg":
            self.update_stored_params_from_avg()

    def accelerate(self):
        for group in self.param_groups:
            if group.get("method") is None:
                continue
            xs = list(group["stored_params"])
            if len(xs) < group["k"]:
                raise ValueError("Not enough stored values to accelerate")
            U = difference_matrix(xs)
            X = torch.vstack(xs[1:])
            if group["method"] == "RNA":
                group["accelerated_params"] = regularized_RRE(X, U, group["lambda_"])
            elif group["method"] == "RRE":
                group["accelerated_params"] = RRE(X, U)
