from collections import deque
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from warnings import warn


def difference_matrix(X):
    return torch.hstack([(x2 - x1)[:, None] for x1, x2 in zip(X[:-1], X[1:])])


class ExperimentBase:
    def __init__(self, seq, f, k, values=None, device="cpu"):
        self.seq = seq
        if values is None:
            self.values = [f(x).item() for x in seq]
        else:
            self.values = values
        self.f = f
        self.k = k
        self.device = device
        self.logs = {}
        self.value_logs = {}
        self.stride = 1

    def plot_values(self, methods=None, n=None, ylim=None, **kwargs):
        if methods is None:
            methods = self.logs.keys()
        s1 = self.values[self.k + 2:]
        fig, ax = plt.subplots(nrows=1, ncols=1, **kwargs)
        if n is None:
            n = len(s1)
        x = np.arange(n) + self.k + 2
        ax.plot(x, s1[:n], label="Original", alpha=0.8)
        for m in methods:
            ax.plot(x[::self.stride], self.value_logs[m][:len(x[::self.stride])], label=m, alpha=0.8)
        ax.legend()
        if ylim is not None:
            ax.set_ylim(*ylim)

    def plot_log_diff(self, methods=None, n=None, ylim=None, vs_original=False, **kwargs):
        best = self.values[-1]
        if not vs_original:
            for s in self.value_logs.values():
                if s[-1] < best:
                    best = s[-1]

        if methods is None:
            methods = self.logs.keys()
        s1 = self.values[self.k + 2:]
        fig, ax = plt.subplots(nrows=1, ncols=1, **kwargs)
        if n is None:
            n = len(s1)
        x = np.arange(n) + self.k + 2
        ax.plot(x, np.log10(np.abs(np.array(s1[:n]) - best)), label="Original", alpha=0.8)
        for m in methods:
            ax.plot(x[::self.stride], np.log10(np.abs(np.array(self.value_logs[m][:len(x[::self.stride])]) - best)),
                    label=m,
                    alpha=0.8)
        ax.legend()
        if ylim is not None:
            ax.set_ylim(*ylim)

    @property
    def best_x(self):
        best = self.values[-1]
        best_x = self.seq[-1]
        for k in self.value_logs.keys():
            if self.value_logs[k][-1] < best:
                best = self.value_logs[k][-1]
                best_x = self.logs[k][-1]
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


class Experiment(ExperimentBase):
    def run_method(self, name, method_f, n=None, method_kwargs=None):
        with torch.no_grad():
            if method_kwargs is None:
                method_kwargs = {}
            S = deque([x.to(self.device) for x in self.seq[:self.k + 1]], maxlen=self.k + 1)
            U = difference_matrix(self.seq[:self.k + 2])
            U = U.to(self.device)
            Ul = deque([U[:, [i]] for i in range(self.k + 1)], maxlen=self.k + 1)
            r = method_f(torch.vstack(list(S)), U, objective=self.f, **method_kwargs).cpu()
            self.logs[name] = [r]
            self.value_logs[name] = [self.f(r).item()]
            old_x = self.seq[self.k + 2].to(self.device)
            if n is None:
                n = len(self.seq)

            for i in range(self.k + 3, n):
                x = self.seq[i].to(self.device)
                S.append(old_x)
                Ul.append((x - old_x)[:, None])
                U = torch.hstack(list(Ul))
                U = U.to(self.device)
                r = method_f(torch.vstack(list(S)), U, objective=self.f, **method_kwargs).cpu()
                self.logs[name].append(r)
                self.value_logs[name].append(self.f(r).item())
                old_x = x


class RestartingExperiment(ExperimentBase):
    def __init__(self, model, k, device="cpu"):
        if device != model.device:
            warn(f"Model and experiment devices don't match. Model device: {model.device}, experiment device: {device}")

        super().__init__(model.log, model.obj, k, values=model.value_log, device=device)
        self.model = deepcopy(model)
        self.model.log = []
        self.model.value_log = []
        self.stride = self.k + 2

    def run_method(self, name, method_f, repeats, method_kwargs=None):
        if method_kwargs is None:
            method_kwargs = {}
        s = self.seq[:self.k + 2]
        with torch.no_grad():
            U = difference_matrix(s).to(self.device)
            st = torch.vstack(list(s[:-1])).to(self.device)
            m = method_f(st, U, objective=self.f, **method_kwargs)
        self.logs[name] = [m.cpu()]
        self.value_logs[name] = [self.f(m).item()]
        for i in range(1, repeats):
            self.model.theta = m
            self.model.fit(-1, max_iter=self.k + 2)
            s = self.model.log
            assert len(s) == self.k + 2, f"{len(s)} != {self.k + 2}"
            with torch.no_grad():
                U = difference_matrix(s).to(self.device)
                st = torch.vstack(list(s[:-1])).to(self.device)
                m = method_f(st, U, objective=self.f, **method_kwargs)
                self.logs[name].append(m.cpu())
                self.value_logs[name].append(self.f(m).item())
                self.model.theta = m
                self.model.log = []
                self.model.value_log = []
