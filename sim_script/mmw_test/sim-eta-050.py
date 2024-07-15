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

log_path = os.path.dirname(os.path.abspath(__file__))
print(log_path)
log = CSV_WRITER_OBJECT(path=log_path)

np.set_printoptions(threshold=10)
np.set_printoptions(linewidth=1000)

from sim_script.mmw_test.config import *

it_list = [200,400,800,1600,3200]

MMW_ETA = 0.05

for it in it_list:
    for seed in range(REPEAT):
        e = env(cell_size=CELL_SIZE,sta_density_per_1m2=RHO,seed=seed)
        bs = binary_search_relaxation()
        bs.force_lower_bound = False
        alg = mmw(nit=it, eta=MMW_ETA)
        # alg.DEBUG=True
        bs.feasibility_check_alg = alg
        z_vec, Z_fin, remainder = bs.run(e.generate_S_Q_hmax())
        bler = e.evaluate_bler(z_vec, Z_fin)
        mbler = np.mean(bler)
        wbler = np.max(bler)
        log.log_mul_scalar(data_name="mmw-"+str(int(it))+"-"+str(int(MMW_ETA*1000))+"-"+str(CELL_SIZE)+"-"+str(int(RHO*10000)),iteration=seed,values=[Z_fin,mbler,wbler]+bler.tolist())

