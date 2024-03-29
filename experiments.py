from collections import deque
from copy import deepcopy
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from extrapolation import difference_matrix


def deque_from_tensors(xs, device, **kwargs):
    return deque([x.to(device) for x in xs], **kwargs)


class ExperimentBase:
    def __init__(self, seq, f, values=None, device="cpu"):
        self.seq = seq
        if values is None:
            self.values = [f(x).item() for x in seq]
        else:
            self.values = values
        self.f = f
        self.device = device
        self.k = {}
        self.stride = {}
        self.logs = {}
        self.value_logs = {}

    def _get_x_axis(self, method, n=None):
        k = self.k[method]
        stride = self.stride.get(method, 1)
        if n is None:
            n = len(self.logs[method])
        else:
            n //= stride
        m = k + 2 + (n - 1) * stride
        x = np.arange(k + 1, m, stride)
        assert len(x) == n, f"{len(x)} != {n}"
        return x

    def plot_values(self, methods=None, n=None, ax=None):
        if methods is None:
            methods = self.logs.keys()
        if ax is None:
            ax = plt.gca()
        if n is None:
            n = len(self.values)
        ax.plot(np.arange(n), self.values[:n], label="Original", alpha=0.8)
        for m in methods:
            x = self._get_x_axis(m, n)
            ax.plot(x, self.value_logs[m][:len(x)], label=m, alpha=0.8)

    def plot_log_diff(self, methods=None, n=None, compare_to="best", ax=None):
        best = self.values[-1]
        if compare_to == "best":
            for s in self.value_logs.values():
                if s[-1] < best:
                    best = s[-1]

        if methods is None:
            methods = self.logs.keys()
        if ax is None:
            ax = plt.gca()
        if n is None:
            n = len(self.values)
        ax.plot(np.arange(n), np.log10(np.abs(np.array(self.values[:n]) - best)), label="Original", alpha=0.8)
        for m in methods:
            x = self._get_x_axis(m, n)
            ax.plot(x,
                    np.log10(np.abs(np.array(self.value_logs[m][:len(x)]) - best)),
                    label=m,
                    alpha=0.8)

    @property
    def best_x(self):
        best = self.values[-1]
        best_x = self.seq[-1]
        for k in self.value_logs.keys():
            idx = np.argmin(self.value_logs[k][-10:])
            if self.value_logs[k][-10:][idx] < best:
                best = self.value_logs[k][-10:][idx]
                best_x = self.logs[k][-10:][idx]
        return best_x

    def save(self, path):
        d = {
            "seq": self.seq,
            "values": self.values,
            "logs": self.logs,
            "value_logs": self.value_logs,
            "k": self.k,
        }
        torch.save(d, path)

    def load(self, path):
        d = torch.load(path)
        self.seq = d["seq"]
        self.values = d["values"]
        self.logs = d["logs"]
        self.value_logs = d["value_logs"]
        self.k = d["k"]

    def value_df(self):
        s = {"Original": pd.Series(self.values)}
        for m, v in self.value_logs.items():
            s[m] = pd.Series(v, index=self._get_x_axis(m))
        return pd.DataFrame(s)


class Experiment(ExperimentBase):
    def run_method(self, name, method_f, k, n=None, method_kwargs=None, input_size=None):
        """Convention (from the Shanks transform):
            X: input_size + 1 samples
            U: size k + 1 (therefore requires k + 2 samples)
        """
        self.k[name] = k
        if input_size is None:
            input_size = k
        with torch.no_grad():
            if method_kwargs is None:
                method_kwargs = {}
            S = deque_from_tensors(self.seq[k - input_size + 1:k + 2], self.device,
                                   maxlen=input_size + 1)
            U = difference_matrix(self.seq[:k + 2])
            U = U.to(self.device)
            # queue of the differences, the indexing guarantees that they're column vectors
            Ul = deque([U[:, [i]] for i in range(k + 1)], maxlen=k + 1)
            r = method_f(torch.vstack(list(S)), U=U, objective=self.f, **method_kwargs)
            self.logs[name] = [r.cpu()]
            self.value_logs[name] = [self.f(r).item()]
            old_x = self.seq[k + 1].to(self.device)  # the last x from the queue
            if n is None:
                n = len(self.seq)

            for i in range(k + 2, n):
                x = self.seq[i].to(self.device)  # the new x
                S.append(x)
                Ul.append((x - old_x)[:, None])
                U = torch.hstack(list(Ul))
                U = U.to(self.device)
                r = method_f(torch.vstack(list(S)), U=U, objective=self.f, **method_kwargs)
                self.logs[name].append(r.cpu())
                self.value_logs[name].append(self.f(r).item())
                old_x = x


class RestartingExperiment(ExperimentBase):
    def __init__(self, model, device="cpu", copy_model=True):
        if device != model.device:
            warn(f"Model and experiment devices don't match. Model device: {model.device}, experiment device: {device}")

        super().__init__(model.log, model.obj, values=model.value_log, device=device)
        if copy_model:
            self.model = deepcopy(model)
        else:
            self.model = model
        self.model.clear_logs()

    def run_method(self, name, method_f, k, repeats, method_kwargs=None, input_size=None):
        if method_kwargs is None:
            method_kwargs = {}
        self.k[name] = k
        self.stride[name] = k + 2
        if input_size is None:
            input_size = k
        s = self.seq[:k + 2]
        with torch.no_grad():
            U = difference_matrix(s).to(self.device)
            st = torch.vstack(list(s[k - input_size + 1:])).to(self.device)
            m = method_f(st, U=U, objective=self.f, **method_kwargs)
        self.logs[name] = [m.cpu()]
        self.value_logs[name] = [self.f(m).item()]
        try:
            for i in range(1, repeats):
                self.model.theta = m
                self.model.run_steps(k + 2)
                s = self.model.log[1:]
                assert len(s) == k + 2, f"{len(s)} != {k + 2}"
                with torch.no_grad():
                    U = difference_matrix(s).to(self.device)
                    st = torch.vstack(list(s[k - input_size + 1:])).to(self.device)
                    m = method_f(st, U=U, objective=self.f, **method_kwargs)
                self.logs[name].append(m.cpu())
                self.value_logs[name].append(self.f(m).item())
                self.model.clear_logs()
        finally:
            self.model.clear_logs()


class OnlineExperiment(ExperimentBase):
    def __init__(self, model, device="cpu", copy_model=True):
        if device != model.device:
            warn(f"Model and experiment devices don't match. Model device: {model.device}, experiment device: {device}")

        super().__init__(model.y_log, model.obj, values=model.value_log, device=device)
        self.x_seq = model.x_log
        if copy_model:
            self.model = deepcopy(model)
        else:
            self.model = model
        self.model.clear_logs()

    def run_method(self, name, method_f, k, repeats, method_kwargs=None):
        if method_kwargs is None:
            method_kwargs = {}
        self.k[name] = k
        y = deque_from_tensors(self.seq[:k + 1], self.device, maxlen=k + 1)
        x = deque_from_tensors(self.x_seq[1:k + 2], self.device, maxlen=k + 1)
        self.logs[name] = []
        self.value_logs[name] = []
        for i in range(repeats):
            X_mat = torch.vstack(list(x)).to(self.device)
            Y_mat = torch.vstack(list(y)).to(self.device)
            with torch.no_grad():
                new_y = method_f(X_mat, Y_mat, objective=self.f, **method_kwargs)
            y.append(new_y.cpu())
            self.logs[name].append(new_y.cpu())
            self.value_logs[name].append(self.f(new_y).item())
            self.model.theta = new_y.to(self.model.device)
            new_x = self.model.step()[2]
            self.model.clear_logs()  # to save some memory
            x.append(new_x.cpu())
