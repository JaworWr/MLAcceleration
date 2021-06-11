import os
import sys
from collections import defaultdict
from typing import Callable, Tuple

import torch
from torch import nn
from torch.utils import data
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook


class Training:
    def __init__(self, loss_fn: Callable[..., torch.Tensor], device="cpu",
                 val_loss_fn: Callable[..., torch.Tensor] = None, tqdm_mode="stdout"):
        self.loss_fn = loss_fn
        self.device = device
        if val_loss_fn is None:
            self.val_loss_fn = loss_fn
        else:
            self.val_loss_fn = val_loss_fn
        self.training_log = []
        self.validation_logs = defaultdict(lambda: [])
        self.tqdm_mode = tqdm_mode

    def train_epoch(self, model: nn.Module, optimizer: torch.optim.Optimizer, data_loader: data.DataLoader,
                    log=True) -> float:
        model.train()
        loss_sum = 0.
        batches = 0
        loss_mean = float("nan")

        using_tqdm = True
        if self.tqdm_mode == "stdout":
            it = tqdm(data_loader, file=sys.stdout)
        elif self.tqdm_mode == "stderr":
            it = tqdm(data_loader, file=sys.stderr)
        elif self.tqdm_mode == "notebook":
            it = tqdm_notebook(data_loader)
        else:
            using_tqdm = False
            it = iter(data_loader)
        for x, y in it:
            x = x.to(self.device)
            y = y.to(self.device)
            optimizer.zero_grad()
            out = model(x)
            loss = self.loss_fn(out, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            batches += 1
            loss_mean = loss_sum / batches
            if using_tqdm:
                it.set_postfix(loss=f"{loss_mean:.4f}", refresh=True)
        if log:
            self.training_log.append(loss_mean)
        return loss_mean

    def validation(self, model: nn.Module, data_loader: data.DataLoader, log: str = None) -> Tuple[float, float]:
        ok = 0
        loss_sum = 0.
        batches = 0
        model.eval()
        with torch.no_grad():
            for x, y in data_loader:
                x = x.cuda()
                y = y.cuda()
                out = model(x)
                loss_sum += self.val_loss_fn(out, y).item()
                y_pred = out.argmax(1)
                ok += torch.sum(y_pred == y).item()
                batches += len(y)
        accuracy = ok / batches
        loss_mean = loss_sum / batches
        if log is not None:
            self.validation_logs[log + "_accuracy"].append(accuracy)
            self.validation_logs[log + "_loss"].append(loss_mean)
        return accuracy, loss_mean


class Logging:
    def __init__(self, path: str, overwrite=False):
        if not overwrite and os.path.exists(path):
            raise RuntimeError(f"File already exists: {path}")
        self.path = path
        self._log_file = None

    @property
    def log_file(self):
        if self._log_file is None:
            self._log_file = open(self.path, "w")
        return self._log_file

    def log(self, *args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=self.log_file, flush=True)


class EarlyStopping:
    def __init__(self, patience, compare_min=True, min_delta=0.):
        self.patience = patience
        self.no_improvement_rounds = 0
        self.compare_min = compare_min
        self.cur_score = float("inf") if compare_min else float("-inf")
        self.min_delta = min_delta

    def should_stop(self, score):
        diff = score - self.cur_score
        if self.compare_min:
            diff = -diff
        if diff > self.min_delta:
            self.no_improvement_rounds = 0
            self.cur_score = score
            return False
        else:
            self.no_improvement_rounds += 1
            return self.no_improvement_rounds >= self.patience
