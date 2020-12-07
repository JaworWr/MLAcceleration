from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch

def difference_matrix(X):
    return torch.hstack([(x2 - x1)[:, None] for x1, x2 in zip(X[:-1], X[1:])])


class Experiment:
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
            ax.plot(x, self.value_logs[m][:n], label=m, alpha=0.8)
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
            ax.plot(x, np.log10(np.abs(np.array(self.value_logs[m][:n]) - best)), label=m, alpha=0.8)
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
