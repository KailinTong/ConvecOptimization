import numpy as np
import scipy.sparse as ss
import utils
from typing import Tuple, Union
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from numba import jit
from pylops.signalprocessing import Radon2D
import finite_differences as fd


def E(x, lamda):
    return lamda * np.sum(
        np.sqrt(((D @ x) ** 2).reshape(2, -1).sum(0))
    ) + np.sum((A @ x - b) ** 2) / 2


def step_size() -> Tuple[float, float, float]:
    '''
    TODO: implement
    You may take any arguments you need

    The step size for `z` and `y` are the same, so we return sigma_1 twice
    tau and sigma are scalars
    '''
    tau = 1
    sigma = 2
    return tau, sigma, sigma


def step_size_prec() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    TODO: implement
    You may take any arguments you need

    Here you should return a tuple of ndarrays, representing
        tau: the step sizes for the primal variable x
        sigma_1: the step sizes for the dual variable y
        sigma_s: the step sizes for the dual variable z
    '''
    tau = np.ones(M * N)
    sigma_1 = np.ones(2 * M * N)
    sigma_2 = np.ones(M * N)
    return tau, sigma_1, sigma_2


def pdhg(
    A: utils.AsMatrix,
    D: ss.coo_matrix,
    b: np.ndarray,
    lamda: float,
    tau: Union[np.ndarray, float],
    sigma_1: Union[np.ndarray, float],
    sigma_2: Union[np.ndarray, float],
):
    x = np.zeros_like(A.T @ b)
    y = np.zeros_like(D @ x)
    z = np.zeros_like(b)

    energy = np.zeros((max_iter,))

    for i in range(max_iter):
        '''
        TODO: implement
        '''

        energy[i] = E(x, lamda)

        if i % 10 == 0:
            print(f'{i=:04d}, {energy[i]=:.4f}')

    return x, energy


def visualize(
    energies: Tuple[np.ndarray, np.ndarray],
    reconstructions: Tuple[np.ndarray, np.ndarray],
    target: np.ndarray,
):
    plt.rc('text', usetex=True)
    _, ax_en = plt.subplots()
    ax_en.loglog(
        energies[0], label='\\( \\tau\\sigma||\\mathcal{K}||^2 < 1 \\)')
    ax_en.loglog(
        energies[1], label='Preconditioned')
    ax_en.legend()
    _, ax_im = plt.subplots(1, 3)
    ax_im[0].imshow(target)
    ax_im[1].imshow(reconstructions[0])
    ax_im[2].imshow(reconstructions[1])
    plt.show()


if __name__ == '__main__':
    # load the data
    target = resize(
        shepp_logan_phantom(), (128, 128), order=3).astype(np.float32)
    M, N = target.shape
    target = target.ravel()

    # scanner geometry
    L = M
    theta = np.linspace(0., 180., L, endpoint=False)

    @jit(nopython=True)
    def radoncurve(x, r, theta):
        return (r - M // 2) / (np.sin(np.deg2rad(theta)) + 1e-15) + \
            np.tan(np.deg2rad(90 - theta)) * x + M // 2

    RLop = Radon2D(
        np.arange(M), np.arange(M), theta, kind=radoncurve, centeredh=True,
        engine='numba', dtype='float32'
    )

    # Convenience class to be able to write A @ x
    A = utils.AsMatrix(
        lambda x: RLop.H * x,
        lambda y: RLop * y,
    )

    # compute the observation
    b = A @ target
    # add 1% noise
    sigma = b.max() * 0.01
    b += sigma * np.random.default_rng().standard_normal(
        *b.shape).astype(b.dtype)
    D = fd.D(M, M)

    max_iter = 500
    lamda = 1.0
    '''
    TODO: call the step size functions with any arguments you need
    '''
    reco, energy = pdhg(
        A, D, b, lamda, *step_size())
    reco_prec, energy_prec = pdhg(
        A, D, b, lamda, *step_size_prec())
    visualize(
        (energy, energy_prec),
        (reco.reshape(M, N), reco_prec.reshape(M, N)),
        target.reshape(M, N)
    )
