import math

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

def csr_scal_rows_inplace(csr: scipy.sparse.csr_matrix, factor):
    nnz_per_row = np.diff(csr.indptr)
    factor = np.repeat(factor, nnz_per_row)
    csr.data = csr.data * factor
    return csr

def csr_scal_cons_inplace(csr, factor:float):
    assert factor != 0
    csr.data = csr.data * factor
    return csr

def csr_expm_rank_dsketch(csr, K, d, r=10):
    d = int(d)
    res = np.zeros((K,d))
    randv = np.random.randn(K,d)/math.sqrt(d)
    res += randv
    for rr in range(r):
        k = rr+1
        randv = (csr @ randv) / k
        res += randv
    return res

def csr_expm_rankd_sketch_autonorm(csr, K, d=10, r=10, normed_it_factor=1., first_it=False, computation_scaling_factor = 1.):
    if first_it:
        normed_it_factor = 1.
    K = int(K)
    d = int(d)
    r = int(r)
    res = np.zeros((K,d))
    randv = np.random.randn(K,d)/math.sqrt(d)
    res += randv
    # csr.data = csr.data * computation_scaling_factor
    for rr in range(r):
        k = rr+1
        randv = (csr @ randv) / k
        res = res / normed_it_factor * computation_scaling_factor + randv
    return res

