import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

LF = 1e32

def get_data() -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv('./winequality-white.csv', sep=';')
    y = data['quality'].to_numpy()
    X = data.to_numpy()
    X[:, -1] = 1
    return X, y


def plot_results(
    ax: list[plt.Axes],
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    E: np.ndarray,
    MAE: np.ndarray
) -> None:
    ax[0].loglog(E)
    ax[0].grid()
    ax[0].set_title('Energy')
    ax[1].semilogx(MAE)
    ax[1].grid()
    ax[1].set_title('Mean Absolute Error')


def get_h(z: float, yi: float, epsilon: float)->float:
    return max([np.abs(z - yi) - epsilon, 0])

def get_step_size(g: np.ndarray, k: int)->float:
    g_norm = np.linalg.norm(g)
    if g_norm > 1/LF:
        return 1 / (g_norm * (k+1)**0.5)
    else:
        return 1 / LF

def subgradient_h(z: float, y_i: float, epsilon: float):
    if z > y_i + epsilon: return 1
    if z < y_i - epsilon: return -1
    if epsilon > np.abs(z - y_i): return 0
    if z == y_i + epsilon: return 1
    if z == y_i - epsilon: return -1
    print('Something wrong!!!')
    return np.nan

def subgradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    w_0: np.ndarray,
    zeta: float,
    epsilon: float,
    maxiter: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # TODO: Implement

    N = X.shape[0]
    L = X.shape[1]

    w = np.expand_dims(w_0, axis=1)
    Q = np.eye(L)
    Q[-1, -1] = 0

    E = np.zeros((maxiter,))
    MAE = np.zeros((maxiter,))


    for k in range(0, maxiter):
        if k % (maxiter / 100) == 0:
            print('zeta {1}, epsilon {2}: {0}%'.format(int(k / maxiter * 100), zeta, epsilon))
        Xw = X @ w
        E[k] = zeta / 2 * w.T @ Q @ w + np.sum([get_h(Xw[i].item(), y[i].item(), epsilon) for i in range(N)])
        MAE[k] = 1 / N * np.sum([np.abs(Xw[i] - y[i]) for i in range(N)])
        g = zeta * Q @ w
        # g += np.sum([X[i] * subgradient_h(Xw[i].item(), y[i], epsilon) for i in range(N)])
        g += np.expand_dims(np.sum([X[i] * subgradient_h(Xw[i].item(), y[i], epsilon) for i in range(N)], axis=0), axis=1)
        t = get_step_size(g, k)
        w = w - t * g

    return w, E, MAE


def primal_dual(
    X: np.ndarray,
    y: np.ndarray,
    w_0: np.ndarray,
    zeta: float,
    epsilon: float,
    maxiter: int = 10_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # TODO: Implement
    E = np.zeros((maxiter,))
    MAE = np.zeros((maxiter,))
    N = X.shape[0]
    L = X.shape[1]

    Q = np.eye(L)
    Q[-1, -1] = 0
    w = np.expand_dims(w_0, axis=1)
    v = np.zeros((N,1))
    tau = sigma = 1 / np.linalg.norm(X)
    yexp = np.expand_dims(y, 1)

    for k in range(0, maxiter):
        if k % (maxiter / 100) == 0:
            print('zeta {1}, epsilon {2}: {0}%'.format(int(k / maxiter * 100), zeta, epsilon))
        Xw = X @ w
        E[k] = zeta / 2 * w.T @ Q @ w + np.sum([get_h(Xw[i].item(), y[i].item(), epsilon) for i in range(N)])
        MAE[k] = 1 / N * np.sum([np.abs(Xw[i] - y[i]) for i in range(N)])

        w_temp = w - tau * X.T @ v
        w_next = np.linalg.inv(np.eye(len(Q)) + tau * zeta * Q) @ w_temp

        v_temp = v + sigma * X @ (2 * w_next - w)
        v = np.clip(np.sign(v_temp - sigma * yexp) * np.maximum(np.abs(v_temp - sigma * yexp) - sigma * zeta, 0), -1, 1)
        w = w_next
    return w, E, MAE


if __name__ == '__main__':
    X, y = get_data()
    # To test the algorithm, only a small data set is used,.
    X = X[0:400,:]
    y = y[0:400]
    w_0: np.ndarray = np.load('w.npy')
    for zeta in [1e0, 1e1, 1e3]:
        for epsilon in [0.2, 1, 5]:
            w, E, MAE = subgradient_descent(X, y, w_0, zeta, epsilon)
            fig, ax = plt.subplots(1, 2)
            plot_results(ax, X, y, w, E, MAE)
            plt.savefig('subgr_z{0}_e{1}'.format(zeta, epsilon).replace('.', '_') + '.png')

            w, E, MAE = primal_dual(X, y, w_0, zeta, epsilon)
            fig, ax = plt.subplots(1, 2)
            plot_results(ax, X, y, w, E, MAE)
            plt.savefig('pd_z{0}_e{1}'.format(zeta, epsilon).replace('.', '_') + '.png')
            plt.show()
