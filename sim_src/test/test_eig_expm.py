import math
import time

from sim_src.alg.mmwm_scipy import mmwm_scipy
from sim_src.env.env import env
from sim_src.scipy_util import csr_expm_rank_dsketch, csr_zero_rows_inplace
import numpy as np

import scipy
from pylanczos import PyLanczos, Exponentiator

from sim_src.util import profile

np.set_printoptions(precision=1)


e = env(cell_size=5,seed=int(time.time()))
a = mmwm_scipy(e.n_sta,e.min_sinr)
a.set_st(e.rxpr)



# def compute_back_diff(eigval,eigvec,orign):
#     tmp = eigvec @ np.diag(eigval) @ eigvec.T
#     diff = tmp - orign
#     diff_sum = np.sum(np.abs(diff))
#     return diff_sum

# @profile
def eig_exp(eigval,eigvec,orign):
    sign = np.sign(eigval)
    eigval[eigval<0] *= -1.
    print(eigval)
    eigval_exp = np.exp(eigval)
    right = eigvec*sign
    tmp_exp = eigvec @ np.diag(eigval_exp) @ right.T
    zzz_exp = eigvec @ np.diag(eigval_exp) @ eigvec.T
    org_exp = scipy.linalg.expm(orign)
    print(tmp_exp[0:5,0:5] + np.eye(5))
    print(zzz_exp[0:5,0:5])
    print(org_exp[0:5,0:5])
    print(eigvec.T @ eigvec)
    return

def eig_exp_max_min(eigval_h, eigvec_h, eigval_l, eigvec_l, orign):
    eigval_h_exp = np.exp(eigval_h)
    h_exp = eigvec_h @ np.diag(eigval_h_exp) @ eigvec_h.T
    eigval_l_exp = np.exp(eigval_l)
    l_exp = eigvec_l @ np.diag(eigval_l_exp) @ eigvec_l.T
    tmp = h_exp + l_exp
    print(tmp[0:5,0:5])
    print(scipy.linalg.expm(orign)[0:5,0:5])
    return
@profile
def test_eig_expm(HH,n=1):
    HH_d = HH.todense()
    eigenvalues_h, eigenvectors_h = scipy.sparse.linalg.eigsh(HH, k=n, which ='LM')
    eigenvalues_l, eigenvectors_l = scipy.sparse.linalg.eigsh(HH, k=n, which ='LM')
    eig_exp_max_min(eigenvalues_h,eigenvectors_h,eigenvalues_l,eigenvectors_l,HH_d)
    engine = Exponentiator(HH)
    output = engine.run(1., np.random.randn(HH.shape[0]))
    engine = PyLanczos(HH, True, n)  # Find 2 maximum eigenpairs
    eigenvalues, eigenvectors = engine.run()

H = a.H.copy()
idx = np.random.randn(e.n_sta)
idx = idx > 0
# csr_zero_rows_inplace(H, idx)
H = H + H.transpose()
print(H.todense()[0:10,0:10])
eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(H, k=1, which ='LM')
H = H/np.abs(eigenvalues[0]) + scipy.sparse.diags(np.ones(e.n_sta)*0.1)
test_eig_expm(H,n=10)
print(np.random.randn(e.n_sta).shape)