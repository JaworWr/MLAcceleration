from collections import deque
from copy import deepcopy

import torch
from torch.optim import SGD

from extrapolation import difference_matrix, regularized_RRE


def params_to_vector(parameters):
    param_vectors = []
    for param in parameters:
        param_vectors.append(param.data.flatten().cpu())
    return deepcopy(torch.hstack(param_vectors))


def params_from_vector(parameters, x):
    idx = 0
    for param in parameters:
        n = param.data.numel()
        param.data[:] = x[idx:idx + n].view(param.data.shape)
        idx += n


class AcceleratedSGD(SGD):

    def __init__(self, params, lr: float, k: int = 10, lambda_: float = 1e-10, momentum: float = 0,
                 dampening: float = 0, weight_decay: float = 0, nesterov: bool = False):
        self.k = k
        self.lambda_ = lambda_
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)

    def add_param_group(self, param_group: dict):
        super().add_param_group(param_group)
        self.reset_stored_params()

    def reset_stored_params(self):
        for group in self.param_groups:
            group["stored_params"] = deque([], maxlen=self.k)
            group.pop("accelerated_params", None)

    def update_stored_params(self):
        for group in self.param_groups:
            x = params_to_vector(group["params"])
            group["stored_params"].append(x)

    def store_parameters(self, target_groups=None):
        if target_groups is None:
            for group in self.param_groups:
                if "accelerated_params" in group:
                    params_from_vector(group["params"], group["accelerated_params"])
        else:
            for group, target in zip(self.param_groups, target_groups):
                if "accelerated_params" in group:
                    params_from_vector(target, group["accelerated_params"])

    def step(self, closure=None):
        super().step(closure)
        self.update_stored_params()

    def accelerate(self):
        for group in self.param_groups:
            xs = list(group["stored_params"])
            if len(xs) < self.k:
                raise ValueError("Not enough stored values to accelerate")
            U = difference_matrix(xs)
            X = torch.vstack(xs[1:])
            group["accelerated_params"] = regularized_RRE(X, U, self.lambda_)
