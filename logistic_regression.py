import torch


def logistic_regression_objective(X, y, tau):
    """
    Get logistic regression objective function
    :param X: Design matrix (size: m x n)
    :param y: Response variable (size: m)
    :param tau: Regularization parameter
    :return: Objective function
    """

    def f(theta):
        loss = torch.log(1 + torch.exp(-y * (X @ theta)))
        reg = 0.5 * tau * torch.sum(theta ** 2)
        return torch.sum(loss) + reg

    return f


def gd_step(f, x, alpha):
    y = f(x)
    y.backward()
    g = x.grad
    with torch.no_grad():
        return x - alpha * g, y


class LogisticRegression:
    def __init__(self, X, y, tau, theta0=None, device="cpu"):
        self.obj = logistic_regression_objective(X, y, tau)
        if theta0 is None:
            self.theta = torch.zeros(X.shape[1], device=device, dtype=X.dtype)
        else:
            self.theta = torch.tensor(theta0, device=device, dtype=X.dtype)
        L = torch.sum(X ** 2).item() / 4. + tau
        self.alpha = 2 / (L + tau)
        self.log = [self.theta.cpu().detach()]
        self.value_log = []
        self.grad_log = []

    def fit(self, eps, max_iter=10000):
        old_theta = None
        iter_ = 0
        while old_theta is None or torch.max(torch.abs(self.theta - old_theta)) > eps:
            old_theta = self.theta
            old_theta.requires_grad_(True)
            res, y = gd_step(self.obj, old_theta, self.alpha)
            self.theta = res.detach()
            self.log.append(self.theta.cpu())
            self.value_log.append(y.item())
            self.grad_log.append(old_theta.grad.detach().cpu())
            iter_ += 1
            if iter_ >= max_iter:
                break
        self.value_log.append(self.obj(self.theta).item())

    def predict(self, X):
        with torch.no_grad():
            scores = 1. / (1. + torch.exp(-X @ self.theta))
            return torch.where(scores > 0.5, 1, -1)
