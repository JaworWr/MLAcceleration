import numpy as np
import torch


def MPE(X, U, qr=True, objective=None):
    # X: (k, n)
    n, k = U.shape
    c = torch.ones(k, device=U.device, dtype=U.dtype)
    A = U[:, :-1]
    b = -U[:, [-1]]
    if qr:
        Q, R = torch.qr(A, some=True)
        c[:-1] = torch.triangular_solve(Q.T @ b, R, upper=True).solution.flatten()
    else:
        M = A.T @ A
        c[:-1] = torch.solve(A.T @ b, M).solution.flatten()
    gamma = c / torch.sum(c)
    return gamma @ X


def RRE(X, U, qr=True, objective=None):
    n, k = U.shape
    b = torch.ones((k, 1), device=U.device, dtype=U.dtype)
    if qr:
        Q, R = torch.qr(U, some=True)
        y = torch.triangular_solve(b, R.T, upper=False).solution
        c = torch.triangular_solve(y, R, upper=True).solution
    else:
        M = U.T @ U
        # M = M / torch.sqrt(torch.sum(M ** 2))
        c = torch.solve(b, M).solution
    gamma = c / torch.sum(c)
    return (gamma.T @ X).flatten()


def regularized_RRE(X, U, lambda_, objective=None):
    n, k = U.shape
    M = U.T @ U
    M = M / torch.sqrt(torch.sum(M ** 2))
    I = torch.eye(k, device=U.device, dtype=U.dtype)
    b = torch.ones((k, 1), device=U.device, dtype=U.dtype)
    c = torch.solve(b, M + lambda_ * I).solution
    gamma = c / torch.sum(c)
    return (gamma.T @ X).flatten()


def RNA(X, U, objective, lambda_range, linesearch=True, normalize=True):
    n, k = U.shape
    solutions = []
    M = U.T @ U
    if normalize:
        M = M / torch.sqrt(torch.sum(M ** 2))
    I = torch.eye(k, device=U.device, dtype=U.dtype)
    b = torch.ones((k, 1), device=U.device, dtype=U.dtype)
    for lambda_ in np.geomspace(lambda_range[0], lambda_range[1], k):
        c = torch.solve(b, M + lambda_ * I).solution
        gamma = c.T / torch.sum(c)
        solutions.append((gamma @ X).flatten())
    values = [objective(x).item() for x in solutions]
    idx = np.argmin(values)
    solution = solutions[idx]

    if linesearch:
        t = 1
        x0 = X[0]
        ft = objective(x0 + t * (solution - x0)).item()
        f2t = objective(x0 + 2 * t * (solution - x0)).item()
        while f2t < ft:
            t *= 2
            ft = f2t
            f2t = objective(x0 + 2 * t * (solution - x0)).item()
        return x0 + t * (solution - x0)
    else:
        return solution


def RNA_cholesky(X, U, objective, lambda_range, linesearch=True):
    n, k = U.shape
    solutions = []
    b = torch.ones((k, 1), device=U.device, dtype=U.dtype)
    for lambda_ in np.geomspace(lambda_range[0], lambda_range[1], k):
        L = torch.zeros((k, k), device=U.device, dtype=U.dtype)
        L[0, 0] = U[:, 0] @ U[:, 0] + lambda_
        for i in range(1, k):
            a = torch.triangular_solve(U[:, :i].T @ U[:, [i]], L[:i, :i], upper=False).solution
            d = torch.sqrt(U[:, i] @ U[:, i] + lambda_ - a.T @ a).item()
            L[i, :i] = a.flatten()
            L[i, i] = d
        y = torch.triangular_solve(b, L, upper=False).solution
        c = torch.triangular_solve(y, L.T, upper=True).solution
        gamma = c.T / torch.sum(c)
        solutions.append((gamma @ X).flatten())
    values = [objective(x).item() for x in solutions]
    idx = np.argmin(values)
    solution = solutions[idx]

    if linesearch:
        t = 1
        x0 = X[0]
        ft = objective(x0 + t * (solution - x0)).item()
        f2t = objective(x0 + 2 * t * (solution - x0)).item()
        while f2t < ft:
            t *= 2
            ft = f2t
            f2t = objective(x0 + 2 * t * (solution - x0)).item()
        return x0 + t * (solution - x0)
    else:
        return solution
