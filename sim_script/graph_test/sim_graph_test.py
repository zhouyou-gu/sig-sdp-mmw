import os

import math
import time

import numpy as np
import scipy.sparse.linalg

from sim_src.alg.binary_search_relaxation import binary_search_relaxation
from sim_src.alg.mmw import mmw
from sim_src.env.env import env
from sim_src.util import *

log_path = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)
print(log_path)
log = CSV_WRITER_OBJECT(path=log_path)

np.set_printoptions(threshold=10)
np.set_printoptions(linewidth=1000)

test_celsize = [int(i) for i in range(5, 16)]
test_density = [float(i)*1e-3 for i in [5,7.5,10]]


seed = 0
repeat = 100
for p in test_density:
    for d in test_celsize:
        for i in range(repeat):
            e = env(cell_size=d,sta_density_per_1m2=p,seed=seed)

            S, Q, _ = e.generate_S_Q_hmax()
            S.setdiag(0)
            S.eliminate_zeros()
            S.sort_indices()
            non_zero_per_row = np.diff(S.indptr)
            max_non_zero_per_row_gain_in = non_zero_per_row.max()

            S_T = S.transpose().tocsr()
            S_T.sort_indices()
            non_zero_per_row = np.diff(S_T.indptr)
            max_non_zero_per_row_gain_out = non_zero_per_row.max()

            non_zero_per_row = np.diff(Q.indptr)
            max_non_zero_per_row_asso = non_zero_per_row.max()

            S_T_S = S_T + S
            non_zero_per_row = np.diff(S_T_S.indptr)
            max_non_zero_per_row_S_T_S = non_zero_per_row.max()

            out = [e.n_sta,d,p,Q.nnz + e.n_sta*2,max_non_zero_per_row_gain_in,max_non_zero_per_row_gain_out,max_non_zero_per_row_asso,max_non_zero_per_row_S_T_S]

            log.log_mul_scalar(data_name="graph_test",iteration=seed,values=out)
            seed += 1
