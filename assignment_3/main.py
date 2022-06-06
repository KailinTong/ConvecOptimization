import numpy as np
import imageio
import matplotlib.pyplot as plt
import finite_differences as fd
from numpy import linalg as LA


def energy_primal(u: np.ndarray) -> float:
    Du = D @ u
    Du = Du.reshape(6, -1)
    ep = lamda * np.sum(np.sqrt(np.sum(Du ** 2, 0, keepdims=True))) + 0.5 * LA.norm(u - u_0) ** 2
    # ep = lamda * np.sum(LA.norm(Du, axis=0)) + 0.5 * LA.norm(u - u_0) ** 2
    return ep


def energy_dual(p: np.ndarray) -> float:
    p_ = p.reshape(6, -1)
    p_norm = np.sqrt(np.sum(p_ ** 2, 0, keepdims=True))
    ig = np.inf
    if np.all(p_norm <= lamda + 0.01):
        ig = 0
    ed = -0.5 * LA.norm(D.T @ p) ** 2 + (D.T @ p) @ u_0 - ig
    return ed


def projc(z: np.ndarray) -> np.ndarray:
    z_ = z.reshape(6, -1)
    z_norm = LA.norm(z_, axis=0)
    coeff = np.clip(lamda / z_norm, 0, 1)
    p = (coeff * z_).T.reshape(-1)
    return p

# Copy from Martin
def proj_dual(p):
    p_r = p.reshape(2 * c, -1)
    return (lamda * p_r / np.maximum(
        np.sqrt(np.sum(p_r ** 2, 0, keepdims=True)),
        lamda
    )).ravel()

def pgm():
    p = np.zeros((2 * m * n * c,))
    # p = 0.01 * np.random.randn(2 * m * n * c)
    ep = np.zeros((num_iter,))
    ed = np.zeros((num_iter,))
    tau = 1 / L * 0.5
    for i in range(num_iter):
        ep[i] = energy_primal(u_0 - D.T @ p)
        ed[i] = energy_dual(p)
        if i % 10 == 0:
            print(f'{i:4d}: dual energy={ed[i]:.3f}')
            print(f'{i:4d}: primal energy={ep[i]:.3f}')
        # p = projc(p - tau * (D @ D.T @ p - D @ u_0))
        p = proj_dual(p - tau * (D @ D.T @ p - D @ u_0))
    u_ = (u_0 - D.T @ p).reshape(c, m, n).transpose(1, 2, 0)
    p_ = p.reshape(c * 2, m, n).transpose(1, 2, 0)
    return u_, p_, ep, ed


def fista():
    p = np.zeros((2 * m * n * c,))
    # p = 0.01 * np.random.randn(2 * m * n * c)

    p_old = p.copy()
    tau = 1 / L * 0.5
    # mu_f = 0
    # mu_g = 0
    # mu = mu_f + mu_g
    # q = tau * mu / (1 + tau * mu_g)
    t_old = 0
    # t = (1 - q * t_old ** 2 + np.sqrt((1 - q * t_old ** 2) ** 2 + 4 * t_old ** 2)) / 2
    t = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
    ep = np.zeros((num_iter,))
    ed = np.zeros((num_iter,))
    # beta = 0
    for i in range(num_iter):
        ep[i] = energy_primal(u_0 - D.T @ p)
        ed[i] = energy_dual(p)
        if i % 10 == 0:
            print(f'{i:4d}: dual energy={ed[i]:.3f}')
            print(f'{i:4d}: primal energy={ep[i]:.3f}')
            # print('beta is {}'.format(beta))
        # beta = (t_old - 1) / t * (1 + tau*mu_g - t*tau*mu) / (1 - tau*mu_f)
        beta = (t_old - 1) / t
        y = p + beta * (p - p_old)
        p_old = p.copy()
        p = proj_dual(y - tau * (D @ D.T @ y - D @ u_0))

        t_old = t
        # t = (1 - q * t_old ** 2 + np.sqrt((1 - q * t_old ** 2) ** 2 + 4 * t_old ** 2)) / 2
        t = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2

    u_ = (u_0 - D.T @ p).reshape(c, m, n).transpose(1, 2, 0)
    p_ = p.reshape(c * 2, m, n).transpose(1, 2, 0)
    return u_, p_, ep, ed


def visualize(
        u_0,
        u_pgm, p_pgm, ep_pgm, ed_pgm,
        u_fista, p_fista, ep_fista, ed_fista,
):
    fig, ax = plt.subplots(2, 2)

    ax[0, 0].imshow(u_0)
    ax[0, 1].imshow(target)
    ax[1, 0].imshow(u_pgm)
    ax[1, 1].imshow(u_fista)
    plt.savefig('mars_lamda_' + str(lamda) + ".png")

    fig, ax = plt.subplots(1, 2)
    ax[0].loglog(ep_pgm, label='PGM')
    ax[0].loglog(ep_fista, label='FISTA')
    ax[0].grid()
    ax[0].legend()
    ax[0].set_title('Energy Prime')
    ax[1].loglog(ed_pgm, label='PGM')
    ax[1].loglog(ed_fista, label='FISTA')
    ax[1].grid()
    ax[1].legend()
    ax[1].set_title('Energy Dual')
    plt.savefig('energy_lamda_' + str(lamda) + ".png")

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(np.arange(len(ep_pgm)), ep_pgm - ed_pgm, label='PGM')
    ax[0].grid()
    ax[0].legend()
    ax[0].set_title('Dual Gap PGM')
    ax[1].plot(np.arange(len(ep_fista)), ep_fista - ed_fista, label='FISTA')
    ax[1].grid()
    ax[1].legend()
    ax[1].set_title('Dual Gap FISTA')
    plt.savefig('dual_gap_lamda_' + str(lamda) + ".png")


    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(u_pgm[:, :, 0:1])
    ax[1].imshow(u_pgm[:, :, 1:2])
    ax[2].imshow(u_pgm[:, :, 2:3])
    plt.savefig('u structure' + ".png")

    fig, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(p_pgm[:, :, 0:1])
    ax[0, 1].imshow(p_pgm[:, :, 1:2])
    ax[0, 2].imshow(p_pgm[:, :, 2:3])
    ax[1, 0].imshow(p_pgm[:, :, 3:4])
    ax[1, 1].imshow(p_pgm[:, :, 4:5])
    ax[1, 2].imshow(p_pgm[:, :, 5:6])
    plt.savefig('p structure' + ".png")

    plt.show()


if __name__ == '__main__':
    target = imageio.imread('mars.png') / 255.
    m, n, c = target.shape
    u_0 = target + 0.2 * np.random.randn(*target.shape)
    # Construction of D assumes channels first
    u_0 = u_0.transpose(2, 0, 1).ravel()
    D = fd.D(m, n, c)
    L = 8
    lamda_set = [0.25, 1., 2.]
    num_iter = 400
    for lamda in lamda_set:
        u_pgm, p_pgm, ep_pgm, ed_pgm = pgm()
        u_fista, p_fista, ep_fista, ed_fista = fista()
        visualize(
        u_0.reshape(c, m, n).transpose(1, 2, 0),
        u_pgm, p_pgm, ep_pgm, ed_pgm,
        u_fista, u_pgm, ep_fista, ed_fista
        )
