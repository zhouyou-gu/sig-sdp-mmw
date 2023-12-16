import math
import time

from sim_src.alg.mmwm_scipy import mmwm_scipy
from sim_src.env.env import env
from sim_src.scipy_util import csr_expm_rank_dsketch
import numpy as np
import scipy
from pylanczos import PyLanczos

from sim_src.util import profile

e = env(cell_size=5,seed=int(time.time()))
# print(e.min_sinr,e.n_sta,e.rxpr)
a = mmwm_scipy(e.n_sta,e.min_sinr)
a.set_st(e.rxpr_hi)

def compute_back_diff(eigval,eigvec,orign):
    tmp = eigvec @ np.diag(eigval) @ eigvec.T
    diff = tmp - orign
    diff_sum = np.sum(np.abs(diff))
    return diff_sum

@profile
def test_eig(HH,n=1):
    HH_d = HH.todense()
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(HH, k=n, which ='LM')
    print('sp',eigenvalues,eigenvectors[0:4,:],eigenvalues.shape,eigenvectors.shape)
    print('sp',compute_back_diff(eigenvalues,eigenvectors,HH_d))
    engine = PyLanczos(HH, True, n)  # Find 2 maximum eigenpairs
    eigenvalues, eigenvectors = engine.run()
    print('pl',eigenvalues,eigenvectors[0:4,:],eigenvalues.shape,eigenvectors.shape)
    print('pl',compute_back_diff(eigenvalues,eigenvectors,HH_d))
    eigenvalues, eigenvectors = np.linalg.eig(HH_d)
    print('np',eigenvalues[0:n],eigenvectors[0:4,0:n],eigenvalues.shape,eigenvectors.shape)
    print('np',compute_back_diff(eigenvalues[0:n],eigenvectors[:,0:n],HH_d))



H = a.H.copy()
H = H + H.transpose()
eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(H, k=1, which ='LM')
H = H/np.abs(eigenvalues[0])
test_eig(H,10)
