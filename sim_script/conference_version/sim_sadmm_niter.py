import os

import math
import time

import numpy as np
import scipy.sparse.linalg

from sim_src.alg.binary_search_relaxation import binary_search_relaxation
from sim_src.alg.gm import MAX_ASSO, MAX_GAIN, MAX_RAND
from sim_src.alg.lrp import lrp_solver
from sim_src.alg.mmw import mmw
from sim_src.alg.sdp_solver import admm_sdp_solver
from sim_src.env.env import env
from sim_src.util import GLOBAL_PROF_ENABLER, plot_a_array, GET_LOG_PATH_FOR_SIM_SCRIPT, CSV_WRITER_OBJECT

# GLOBAL_PROF_ENABLER.DISABLE()

log_path = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)
print(log_path)
log = CSV_WRITER_OBJECT(path=log_path)

np.set_printoptions(threshold=10)
np.set_printoptions(linewidth=1000)

REPEAT = 20
for RHO in [50e-4,100e-4]:
    for CELL_SIZE in range(5,16):
        g_bound_iter_list = []
        g_bound_time_list = []
        n_bound_iter_list = []
        n_bound_time_list = []
        for seed in range(REPEAT):
            e = env(cell_size=CELL_SIZE,sta_density_per_1m2=RHO,seed=seed)
            bs = binary_search_relaxation()
            alg = admm_sdp_solver(nit=100)
            bs.feasibility_check_alg = alg
            gb_tic = bs._get_tic()
            z_vec, Z_fin, remainder = bs.run(e.generate_S_Q_hmax())
            gb_tim = bs._get_tim(gb_tic)
            d = bs.LOGGED_NP_DATA["bs_search_per_it"].shape[0]
            g_bound_iter_list.append(d)
            g_bound_time_list.append(gb_tim)

            bs = binary_search_relaxation()
            bs.force_full_bound = True
            alg = admm_sdp_solver(nit=100)
            bs.feasibility_check_alg = alg
            nb_tic = bs._get_tic()
            z_vec, Z_fin, remainder = bs.run(e.generate_S_Q_hmax())
            nb_tim = bs._get_tim(nb_tic)
            d = bs.LOGGED_NP_DATA["bs_search_per_it"].shape[0]
            n_bound_iter_list.append(d)
            n_bound_time_list.append(nb_tim)

        log.log_mul_scalar(data_name="g_bound_iter_list-"+str(CELL_SIZE)+"-"+str(int(RHO*10000)),iteration=0,values=g_bound_iter_list)
        log.log_mul_scalar(data_name="g_bound_time_list-"+str(CELL_SIZE)+"-"+str(int(RHO*10000)),iteration=0,values=g_bound_time_list)
        log.log_mul_scalar(data_name="n_bound_iter_list-"+str(CELL_SIZE)+"-"+str(int(RHO*10000)),iteration=0,values=n_bound_iter_list)
        log.log_mul_scalar(data_name="n_bound_time_list-"+str(CELL_SIZE)+"-"+str(int(RHO*10000)),iteration=0,values=n_bound_time_list)
