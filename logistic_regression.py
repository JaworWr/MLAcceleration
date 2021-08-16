import torch
import numpy as np


def gd_step(f, x, alpha):
    y = f(x)
    y.backward()
    g = x.grad
    with torch.no_grad():
        return x - alpha * g, y


def nesterov_step(f, x, alpha, beta):
    val = f(x)
    val.backward()
    g = x.grad
    with torch.no_grad():
        x1 = x - alpha * g
        return x1, (1. + beta) * x1 - beta * x, val


class LogisticRegression:
    def __init__(self, X, y, tau, theta0=None, device="cpu"):
        self.X = X
        self.targets = y
        self.tau = tau
        if theta0 is None:
            self.theta = torch.zeros(X.shape[1], device=device, dtype=X.dtype)
        else:
            self.theta = torch.tensor(theta0, device=device, dtype=X.dtype)
        self.value_log = []
        self.device = device

    def _obj(self, theta):
        t = -self.targets * (self.X @ theta)
        loss = torch.logaddexp(t, torch.zeros_like(t))
        reg = 0.5 * self.tau * torch.sum(theta ** 2)
        return torch.sum(loss) + reg

    @property
    def obj(self):
        return lambda theta: self._obj(theta)

    def step(self):
        raise NotImplementedError

    def run_steps(self, k):
        for _ in range(k):
            self.step()
        self.value_log.append(self.obj(self.theta).item())

    def fit(self, eps, max_iter=10000):
        old_theta = None
        iter_ = 0
        while old_theta is None or torch.max(torch.abs(self.theta - old_theta)) > eps:
            y, old_theta = self.step()[:2]
            iter_ += 1
            if iter_ >= max_iter:
                break
        self.value_log.append(self.obj(self.theta).item())

    def predict(self, X):
        with torch.no_grad():
            scores = 1. / (1. + torch.exp(-X @ self.theta))
            return torch.where(scores > 0.5, 1, -1)

    def to(self, device):
        self.device = device
        self.X = self.X.to(device=device)
        self.targets = self.targets.to(device=device)

    @property
    def x(self):
        raise NotImplementedError

    @property
    def y(self):
        raise NotImplementedError


class LogisticRegressionGD(LogisticRegression):
    def __init__(self, X, y, tau, theta0=None, device="cpu", log_grad=True):
        super().__init__(X, y, tau, theta0, device)
        L = torch.linalg.norm(X.cpu(), 2) ** 2 / 4. + tau
        self.alpha = 2 / (L + tau)
        self.log = [self.theta.cpu().detach()]
        self.grad_log = []
        self.log_grad = log_grad

    def step(self):
        old_theta = self.theta
        old_theta.requires_grad_(True)
        res, y = gd_step(self.obj, old_theta, self.alpha)
        self.theta = res.detach()
        self.log.append(self.theta.cpu())
        self.value_log.append(y.item())
        if self.log_grad:
            self.grad_log.append(old_theta.grad.detach().cpu())
        return y, old_theta

    def clear_logs(self):
        self.log = [self.theta.cpu().detach()]
        self.value_log = []

    @property
    def x_log(self):
        return self.log


class LogisticRegressionNesterov(LogisticRegression):
    def __init__(self, X, y, tau, theta0=None, device="cpu", log_x=True, log_grad=True):
        super().__init__(X, y, tau, theta0, device)
        L = torch.sum(X ** 2).item() / 4. + tau
        self.alpha = 1. / L
        self.beta = (np.sqrt(L) - np.sqrt(tau)) / (np.sqrt(L) + np.sqrt(tau))
        self.x_log = [self.theta.cpu().detach()]
        self.y_log = [self.theta.cpu().detach()]
        self.grad_log = []
        self.log_x = log_x
        self.log_grad = log_grad

    def step(self):
        old_theta = self.theta
        old_theta.requires_grad_(True)
        x, y, val = nesterov_step(self.obj, old_theta, self.alpha, self.beta)
        self.theta = y.detach()
        self.y_log.append(self.theta.cpu())
        if self.log_x:
            self.x_log.append(x.detach().cpu())
        self.value_log.append(val.item())
        if self.log_grad:
            self.grad_log.append(old_theta.grad.detach().cpu())
        return val, old_theta, x.detach()

    def clear_logs(self):
        self.x_log = [self.theta.cpu().detach()]
        self.y_log = [self.theta.cpu().detach()]
        self.value_log = []
