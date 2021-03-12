from collections import deque
from copy import deepcopy

import torch
from torch.optim import SGD

from extrapolation import difference_matrix, regularized_RRE, RRE


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
                 dampening: float = 0, weight_decay: float = 0, nesterov: bool = False,
                 mode: str = "epoch", method: str = "RNA"):
        self.k = k
        self.lambda_ = lambda_
        self.mode = mode
        self.method = method
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)

    def add_param_group(self, param_group: dict):
        super().add_param_group(param_group)
        self.reset_stored_params()

    def reset_stored_params(self):
        for group in self.param_groups:
            group["stored_params"] = deque([], maxlen=self.k)
            group.pop("accelerated_params", None)
            group.pop("stored_params_avg", None)

    def update_stored_params(self):
        for group in self.param_groups:
            x = params_to_vector(group["params"])
            group["stored_params"].append(x)

    def update_param_avg(self):
        for group in self.param_groups:
            x = params_to_vector(group["params"])
            if "stored_params_avg" in group:
                group["stored_params_ctr"] += 1
                c = 1 / group["stored_params_ctr"]
                group["stored_params_avg"] = c * x + (1 - c) * group["stored_params_avg"]
            else:
                group["stored_params_avg"] = x
                group["stored_params_ctr"] = 1

    def update_stored_params_from_avg(self):
        for group in self.param_groups:
            x = group.pop("stored_params_avg", None)
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
            xs = list(group["stored_params"])
            if len(xs) < self.k:
                raise ValueError("Not enough stored values to accelerate")
            U = difference_matrix(xs)
            X = torch.vstack(xs[1:])
            if self.method == "RNA":
                group["accelerated_params"] = regularized_RRE(X, U, self.lambda_)
            else:
                group["accelerated_params"] = RRE(X, U)


class SeparateAcceleratedSGD(AcceleratedSGD):

    def update_stored_params(self):
        for group in self.param_groups:
            params = [p.data.flatten().cpu() for p in group["params"]]
            group["stored_params"].append(params)

    def update_param_avg(self):
        for group in self.param_groups:
            params = [p.data.flatten().cpu() for p in group["params"]]
            if "stored_params_avg" in group:
                group["stored_params_ctr"] += 1
                c = 1 / group["stored_params_ctr"]
                new_avg = []
                for p, p_avg in zip(params, group["stored_params_avg"]):
                    new_avg.append(c * p + (1 - c) * p_avg)
                group["stored_params_avg"] = new_avg
            else:
                group["stored_params_avg"] = params
                group["stored_params_ctr"] = 1

    def store_parameters(self, target_groups=None):
        if target_groups is None:
            for group in self.param_groups:
                if "accelerated_params" in group:
                    for p, p_acc in zip(group["params"], group["accelerated_params"]):
                        p.data[:] = p_acc.view(p.data.shape)
        else:
            for group, target in zip(self.param_groups, target_groups):
                if "accelerated_params" in group:
                    for p, p_acc in zip(target, group["accelerated_params"]):
                        p.data[:] = p_acc.view(p.data.shape)

    def accelerate(self):
        for group in self.param_groups:
            params = list(group["stored_params"])
            if len(params) < self.k:
                raise ValueError("Not enough stored values to accelerate")
            group["accelerated_params"] = []
            for i in range(len(params[0])):
                xs = [ps[i] for ps in params]
                U = difference_matrix(xs)
                X = torch.vstack(xs[1:])
                group["accelerated_params"].append(regularized_RRE(X, U, self.lambda_))
