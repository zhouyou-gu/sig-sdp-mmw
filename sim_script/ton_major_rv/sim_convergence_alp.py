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

REPEAT =  20

RHO =  75e-4

for CELL_SIZE in [10]:
    for ALPHA in [0.25, 0.5, 1, 2, 4]:
        ETA = 0.04
        NIT = math.ceil(1./ETA/ETA)
        for seed in range(REPEAT):
            e = env(cell_size=CELL_SIZE,sta_density_per_1m2=RHO,seed=seed, txp_offset=ALPHA+1)
            bs = binary_search_relaxation()
            bs.force_lower_bound = False
            alg = admm_sdp_solver(nit=1000)
            bs.feasibility_check_alg = alg
            z_vec, Z_fin, remainder = bs.run(e.generate_S_Q_hmax())
            bler = e.evaluate_bler(z_vec, Z_fin)
            mbler = np.mean(bler)
            wbler = np.max(bler)

            alg = mmw(nit=NIT, eta=ETA)
            alg.LOG_GAP = True
            _, gX = alg.run_with_state(0,Z_fin,e.generate_S_Q_hmax())
            ub = alg.LOGGED_NP_DATA["gap"][:,3]
            lb = alg.LOGGED_NP_DATA["gap"][:,4]
            print(ub,lb)
            log.log_mul_scalar(data_name="mmw-dual-"+str(CELL_SIZE)+"-"+str(int(RHO*10000))+"-"+str(int(ETA*100)),iteration=seed,values=ub.tolist())
            log.log_mul_scalar(data_name="mmw-dual-"+str(CELL_SIZE)+"-"+str(int(RHO*10000))+"-"+str(int(ETA*100)),iteration=seed,values=lb.tolist())
