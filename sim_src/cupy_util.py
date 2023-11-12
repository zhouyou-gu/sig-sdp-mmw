import cupy as cp

from sim_src.util import profile
import cupyx
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from cupyx.scipy.sparse import coo_matrix as cp_coo_matrix

@profile
def cupy_csr_expm_rank_dsketch(csr, K, d, r=10):
    cp_csr = cp_coo_matrix(csr)
    print(cp_csr.getnnz())
    res = cp.zeros((K,d))
    randv = cp.random.randn(K,d)
    res += randv
    for rr in range(r):
        k = r+1
        randv = (cp_csr @ randv)
        res = cp.add(res,randv)
    return res
