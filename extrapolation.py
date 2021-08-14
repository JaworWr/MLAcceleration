import numpy as np
import torch


def normalize(x):
    s = torch.sum(x)
    if torch.abs(s) < 1e-10:
        return torch.ones_like(x) / x.shape[0]
    else:
        return x / s


def MPE(X, U, qr=True, objective=None):
    # X: (k, n)
    n, k = U.shape
    c = torch.ones(k, device=U.device, dtype=U.dtype)
    A = U[:, :-1]
    b = -U[:, [-1]]
    if qr:
        Q, R = torch.linalg.qr(A, mode="reduced")
        c[:-1] = torch.triangular_solve(Q.T @ b, R, upper=True).solution.flatten()
    else:
        M = A.T @ A
        c[:-1] = torch.solve(A.T @ b, M).solution.flatten()
    gamma = normalize(c)
    return gamma @ X


def RRE(X, U, qr=True, objective=None):
    n, k = U.shape
    b = torch.ones((k, 1), device=U.device, dtype=U.dtype)
    if qr:
        Q, R = torch.linalg.qr(U, mode="r")
        y = torch.triangular_solve(b, R.T, upper=False).solution
        c = torch.triangular_solve(y, R, upper=True).solution
    else:
        M = U.T @ U
        # M = M / torch.sqrt(torch.sum(M ** 2))
        c = torch.solve(b, M).solution
    gamma = normalize(c)
    return (gamma.T @ X).flatten()


def regularized_RRE(X, U, lambda_, objective=None):
    n, k = U.shape
    M = U.T @ U
    M = M / torch.linalg.norm(M, 2)
    I = torch.eye(k, device=U.device, dtype=U.dtype)
    b = torch.ones((k, 1), device=U.device, dtype=U.dtype)
    c = torch.solve(b, M + lambda_ * I).solution
    gamma = normalize(c)
    return (gamma.T @ X).flatten()


def MMPE(X, U, objective=None):
    n, k = U.shape
    c = torch.ones(k, device=U.device, dtype=U.dtype)
    c[:-1] = torch.solve(-U[:k - 1, [-1]], U[:k - 1, :-1]).solution.flatten()
    gamma = normalize(c)
    return gamma @ X


def TEA_solve(X, U, q=None, objective=None):
    n, k2 = U.shape
    k = k2 // 2
    if q is None:
        q = torch.ones(n, device=U.device, dtype=U.dtype)
    A = torch.zeros((k, k + 1), device=U.device, dtype=U.dtype)
    for i in range(k):
        A[i, :] = q[None, :] @ U[:, i:i + k + 1]
    c = torch.ones(k + 1, device=U.device, dtype=U.dtype)
    c[:-1] = torch.solve(-A[:, [-1]], A[:, :-1]).solution.flatten()
    gamma = normalize(c)
    return gamma @ X


def inv(x):
    return x / torch.sum(x ** 2, 1, keepdim=True)


def vector_epsilon_v1(X, k, U=None, objective=None):
    """Vector epsilon algorithm using Moore–Penrose generalised inverse"""
    e0 = torch.zeros((X.shape[0] + 1, X.shape[1]), device=X.device, dtype=X.dtype)
    e1 = X
    e2 = None

    for _ in range(2 * k):
        e2 = e0[1:-1] + inv(e1[1:] - e1[:-1])
        e0 = e1
        e1 = e2
    return e2.flatten()


def vector_epsilon_v2(X, k, U=None, objective=None, q=None):
    """Vector epsilon algorithm using a scalar product, i.e. the topological epsilon algorithm"""
    n, m = X.shape
    e_odd = torch.zeros((n + 1, m), device=X.device, dtype=X.dtype)
    e_even = X.clone()

    if q is None:
        q = torch.ones(m, device=X.device, dtype=X.dtype)

    for i in range(k):
        for j in range(n - 2 * i - 1):
            e_odd[j] = e_odd[j + 1] + q / (q @ (e_even[j + 1] - e_even[j]))
        for j in range(n - 2 * i - 2):
            e_even[j] = e_even[j + 1] + (e_even[j + 1] - e_even[j]) \
                        / ((e_odd[j + 1] - e_odd[j]) @ (e_even[j + 1] - e_even[j]))
    return e_even[:n - 2 * k].flatten()


def topological_vector_epsilon(X: torch.Tensor, k, U=None, objective=None, q=None):
    """Simplified topological epsilon algorithm"""
    if q is None:
        q = torch.ones(X.shape[1], device=X.device, dtype=X.dtype)
    e = X.clone()
    eps1 = torch.zeros((X.shape[0] + 1, 1))
    eps2 = X @ q[:, None]
    for i in range(k):
        # scalar update for 2k+1
        eps1 = eps1[1:-1] + 1. / (eps2[1:] - eps2[:-1])
        # vector update
        e = e[1:-1] + (e[2:] - e[1:-1]) / ((eps2[2:] - eps2[1:-1]) * (eps1[1:] - eps1[:-1]))
        # scalar update for 2k+2
        eps2 = eps2[1:-1] + 1. / (eps1[1:] - eps1[:-1])
    return e.flatten()


def RNA(X, U, objective, lambda_range, linesearch=True, norm=True):
    n, k = U.shape
    solutions = []
    M = U.T @ U
    if norm:
        M /= torch.linalg.norm(M, 2)
    I = torch.eye(k, device=U.device, dtype=U.dtype)
    b = torch.ones((k, 1), device=U.device, dtype=U.dtype)
    for lambda_ in np.geomspace(lambda_range[0], lambda_range[1], k):
        c = torch.solve(b, M + lambda_ * I).solution
        gamma = normalize(c.T)
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


def RNA_cholesky(X, U, objective, lambda_range, linesearch=True, norm=True):
    n, k = U.shape
    solutions = []
    b = torch.ones((k, 1), device=U.device, dtype=U.dtype)
    if norm:
        U_norm = torch.linalg.norm(U, 2).item()
    else:
        U_norm = 1.
    # \|U^T U\| = \|U\|^2
    for lambda_ in np.geomspace(lambda_range[0], lambda_range[1], k) * U_norm ** 2:
        L = torch.zeros((k, k), device=U.device, dtype=U.dtype)
        L[0, 0] = torch.sqrt(U[:, 0] @ U[:, 0] + lambda_)
        for i in range(1, k):
            a = torch.triangular_solve(U[:, :i].T @ U[:, [i]], L[:i, :i], upper=False).solution
            d = torch.sqrt(U[:, i] @ U[:, i] + lambda_ - a.T @ a).item()
            assert d != 0, f"L will be singular; lambda={lambda_}, i={i}"
            L[i, :i] = a.ravel()
            L[i, i] = d
        L /= U_norm
        y = torch.triangular_solve(b, L, upper=False).solution
        c = torch.triangular_solve(y, L.T, upper=True).solution
        gamma = normalize(c.T)
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


def mixing_RNA(X, Y, lambda_, beta, objective=None):
    X = X.T
    Y = Y.T
    n, k = X.shape
    U = X - Y
    M = U.T @ U
    M = M / torch.linalg.norm(M, 2)
    I = torch.eye(k, device=U.device, dtype=U.dtype)
    b = torch.ones((k, 1), device=U.device, dtype=U.dtype)
    c = torch.solve(b, M + lambda_ * I).solution
    gamma = normalize(c)
    return ((Y - beta * U) @ gamma).flatten()


def optimal_RNA(X, Y, lambda_, alpha, beta, objective, f_xi=None):
    # optimal adaptive algorithm from the paper
    # alpha, beta - step sizes for the Nesterov acceleration
    y_extr = mixing_RNA(X, Y, lambda_, 0)
    z = (y_extr + beta * X[-1]) / (1. + beta)
    if f_xi is None:
        f_xi = objective(X[-1]).item()
    if objective(z).item() < f_xi - 0.5 * alpha * f_xi ** 2:
        return y_extr
    else:
        return (1. + beta) * X[-1] - beta * X[-2]


def difference_matrix(X):
    k = len(X) - 1
    U = torch.empty((X[0].shape[0], k), dtype=X[0].dtype)
    for i in range(k):
        U[:, i] = X[i + 1] - X[i]
    return U


def absmax(x, axis=None):
    idx = np.argmax(np.abs(x), axis=axis)
    if axis is None:
        return x.ravel()[idx]
    else:
        idx = np.expand_dims(idx, axis=axis)
        return np.take_along_axis(x, idx, axis=axis)


def safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


def levin_remainder(x, type, vector):
    N = x.shape[0]
    dx = np.diff(x, axis=0)

    if type == "t":
        r = dx
    elif type == "u":
        if vector:
            r = np.arange(1, N) * absmax(dx, 1).ravel()
        else:
            r = np.arange(1, N)[:, None] * dx
    elif type == "v":
        r = safe_div(dx[:-1] * dx[1:], dx[1:] - dx[:-1])
    else:
        raise RuntimeError("Invalid type")

    if vector and type != "u":
        r = absmax(r, 1).ravel()
    return r


def h_algorithm(xt, k, type="t", U=None, objective=None):
    """Vector E-algorithm"""
    x = xt.cpu().numpy()
    r = levin_remainder(x, type, True)
    N = min(x.shape[0], r.shape[0])
    h = x[:N]
    g = r[None, :N] / (np.arange(N)[None, :] + 1) ** np.arange(k)[:, None]

    for i in range(k):
        h = h[:-1] - safe_div(g[i, :-1, None] * np.diff(h, axis=0), np.diff(g[i], axis=0)[:, None])
        if i < k - 1:
            g = g[:, :-1] - safe_div(g[i, :-1] * np.diff(g, axis=1), np.diff(g[i], axis=0))

    return torch.tensor(h.ravel(), dtype=xt.dtype, device=xt.device)


def e_algorithm(xt, k, type="t", U=None, objective=None):
    """Scalar E-algorithm performed along axis 0"""
    x = xt.cpu().numpy()
    r = levin_remainder(x, type, False)
    N = min(x.shape[0], r.shape[0])
    e = x[:N]
    pow_ = (np.arange(N)[None, :] + 1) ** np.arange(k)[:, None]
    g = r[None, :N, :] / pow_[:, :, None]

    for i in range(k):
        e = e[:-1] - safe_div(g[i, :-1] * np.diff(e, axis=0), np.diff(g[i], axis=0))
        if i < k - 1:
            g = g[:, :-1] - safe_div(g[i, :-1] * np.diff(g, axis=1), np.diff(g[i], axis=0))
    return torch.tensor(e.ravel(), dtype=xt.dtype, device=xt.device)


def j_algorithm(xt, deltas, type="t", k=None, U=None, objective=None):
    def step(s, d):
        n = s.shape[0] - 1
        return np.diff(s, axis=0) / d[:n, None]

    x = xt.cpu().numpy()
    r = levin_remainder(x, type, True)
    N = min(x.shape[0], r.shape[0])
    num = x[:N, :] / r[:N, None]
    denum = 1 / r[:N, None]

    if k is not None:
        deltas = deltas[:k]
    for d in deltas:
        num = step(num, d)
        denum = step(denum, d)
    res = num / denum
    return torch.tensor(res.ravel(), dtype=xt.dtype, device=xt.device)
