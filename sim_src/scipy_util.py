import scipy
import numpy as np

from sim_src.util import profile


def csr_zero_rows_inplace(csr: scipy.sparse.csr_matrix, mask):
    nnz_per_row = np.diff(csr.indptr)
    mask_i = np.ones(mask.shape, dtype=bool)
    mask_i[mask] = False
    mask_e = np.repeat(mask_i, nnz_per_row)
    nnz_per_row[mask] = 0
    csr.data = csr.data[mask_e]
    csr.indices = csr.indices[mask_e]
    csr.indptr[1:] = np.cumsum(nnz_per_row)
    return csr

def csr_scal_cons_inplace(csr, factor:float):
    assert factor != 0
    csr.data = csr.data * factor
    return csr

@profile
def csr_expm_rank_dsketch(csr, K, d, r=10):
    res = np.zeros((K,d))
    randv = np.random.randn(K,d)/d
    res += randv
    for rr in range(r):
        k = r+1
        randv = (csr @ randv) / k
        res += randv
    return res

