import os

import math
import time

import numpy as np
import scipy.sparse.linalg

from sim_src.alg.binary_search_relaxation import binary_search_relaxation
from sim_src.alg.gm import MAX_ASSO, MAX_GAIN, MAX_RAND
from sim_src.alg.lrp import lrp_solver
from sim_src.alg.mmw import mmw
from sim_src.alg.sdp_solver import admm_sdp_solver, rand_sdp_solver
from sim_src.env.env import env
from sim_src.util import GLOBAL_PROF_ENABLER, plot_a_array, GET_LOG_PATH_FOR_SIM_SCRIPT, CSV_WRITER_OBJECT

# GLOBAL_PROF_ENABLER.DISABLE()

log_path = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)
print(log_path)
log = CSV_WRITER_OBJECT(path=log_path)

np.set_printoptions(threshold=10)
np.set_printoptions(linewidth=1000)

REPEAT = 5

RHO =  75e-4

for CELL_SIZE in range(13,16):
    for seed in range(REPEAT):
        e = env(cell_size=CELL_SIZE,sta_density_per_1m2=RHO,seed=seed)

        res = []
        print("MMW")
        bs = binary_search_relaxation()
        tic = bs._get_tic()
        alg = mmw(nit=150,eta=0.04)
        bs.feasibility_check_alg = alg
        z_vec, Z_fin, remainder = bs.run(e.generate_S_Q_hmax())
        d = bs.LOGGED_NP_DATA["bs_search_per_it"].shape[0]
        tim = bs._get_tim(tic)

        res.append(d)
        res.append(tim)

        print("SCS")
        bs = binary_search_relaxation()
        tic = bs._get_tic()
        alg = admm_sdp_solver(nit=100)
        bs.feasibility_check_alg = alg
        z_vec, Z_fin, remainder = bs.run(e.generate_S_Q_hmax())
        d = bs.LOGGED_NP_DATA["bs_search_per_it"].shape[0]
        tim = bs._get_tim(tic)

        res.append(d)
        res.append(tim)

        print("MWM-NB")
        bs = binary_search_relaxation()
        tic = bs._get_tic()
        alg = mmw(nit=150,eta=0.04)
        bs.feasibility_check_alg = alg
        bs.force_full_bound = True
        z_vec, Z_fin, remainder = bs.run(e.generate_S_Q_hmax())
        d = bs.LOGGED_NP_DATA["bs_search_per_it"].shape[0]
        tim = bs._get_tim(tic)

        res.append(d)
        res.append(tim)


        log.log_mul_scalar(data_name="time-"+str(CELL_SIZE)+"-"+str(int(RHO*10000)),iteration=seed,values=res)
