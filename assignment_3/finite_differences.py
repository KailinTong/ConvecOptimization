import scipy.sparse as sp
import numpy as np


def D(
    H: int,
    W: int,
    C: int = 1
) -> sp.coo_matrix:
    row = np.arange(0, H * W)
    dat = np.ones(H * W, dtype=np.float32)
    col = np.arange(0, H * W).reshape(H, W)
    col_xp = np.hstack([col[:, 1:], col[:, -1:]])
    col_yp = np.vstack([col[1:, :], col[-1:, :]])
    FD1 = sp.coo_matrix((dat, (row, col_xp.flatten())), shape=(H * W, H * W)) \
        - sp.coo_matrix((dat, (row, col.flatten())), shape=(H * W, H * W))
    FD2 = sp.coo_matrix((dat, (row, col_yp.flatten())), shape=(H * W, H * W)) \
        - sp.coo_matrix((dat, (row, col.flatten())), shape=(H * W, H * W))
    return sp.block_diag([sp.vstack([FD1, FD2]) for _ in range(C)])
