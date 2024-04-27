import math
import os
from datetime import datetime
import pprint
from time import time

import matplotlib.pyplot as plt


import numpy as np
import torch
def p_true(probability_of_true):
    return np.random.choice([True, False], p=[probability_of_true, 1 - probability_of_true])

def DbToRatio(a):
    return 10.0**(a/10.)

def RatioToDb(a):
    return 10.0 * math.log10(a)

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LONG_TYPE = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor

def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()

def to_tensor(ndarray, requires_grad=False, dtype=FLOAT):
    t = torch.from_numpy(ndarray)
    t.requires_grad_(requires_grad)
    if USE_CUDA:
        return t.type(dtype).to(torch.cuda.current_device())
    else:
        return t.type(dtype)
def to_device(var):
    if USE_CUDA:
        return var.to(torch.cuda.current_device())
    return var

def cat_str_dot_txt(sl):
    ret = ""
    for s in sl:
        ret += s
        ret += "."
    ret += "txt"

    return ret


def soft_update_inplace(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update_inplace(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def add_param_noise_inplace(target, std=0.01):
    for target_param in list(target.parameters()):
        d = np.random.randn(1)
        d = d * std
        d = to_tensor(d, requires_grad=False)
        target_param.data.add_(d)

def get_current_time_str():
    return datetime.now().strftime("%Y-%B-%d-%H-%M-%S")

def counted(f):
    def wrapped(self, *args, **kwargs):
        self.N_STEP += 1
        return f(self, *args, **kwargs)

    return wrapped

def timed(f):
    def wrapped(self, *args):
        ts = time()
        result = f(self, *args)
        te = time()
        print('%s func:%r took: %2.4f sec' % (self, f.__name__, te - ts))
        return result

    return wrapped


from line_profiler import LineProfiler
from functools import wraps

class prof_enabler:
    enabled = True
    def DISABLE(self):
        self.enabled = False
    def ENABLE(self):
        self.enabled = True

GLOBAL_PROF_ENABLER = prof_enabler()

def profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        prof = LineProfiler()
        try:
            return prof(func)(*args, **kwargs)
        finally:
            if GLOBAL_PROF_ENABLER.enabled:
                prof.print_stats(output_unit=1e-6)

    return wrapper

LOGGED_NP_DATA_HEADER_SIZE = 3

class StatusObject:
    N_STEP = 0
    DISABLE_ALL_DEBUG = False
    DEBUG_STEP = 100
    DEBUG = False

    MOVING_AVERAGE_TIME_WINDOW = 100

    INIT_MOVING_AVERAGE = False
    INIT_LOGGED_NP_DATA = False

    MOVING_AVERAGE_DICT = {}
    MOVING_AVERAGE_DICT_N_STEP = {}
    LOGGED_NP_DATA = {}

    LOGGED_CLASS_NAME = None
    def save(self, path: str, postfix: str):
        pass

    def save_np(self, path: str, postfix: str):
        try:
            os.mkdir(path)
        except:
            pass
        for key in self.LOGGED_NP_DATA:
            if self.LOGGED_CLASS_NAME:
                data_name = "%s.%s.%s.txt" % (self.LOGGED_CLASS_NAME,key,postfix)
            else:
                data_name = "%s.%s.%s.txt" % (self.__class__.__name__,key,postfix)
            data_path = os.path.join(path,data_name)
            np.savetxt(data_path, self.LOGGED_NP_DATA[key] , delimiter=',')

    def _add_np_log(self, key, float_row_data, g_step=0):
        if not self.INIT_LOGGED_NP_DATA:
            self.LOGGED_NP_DATA = {}
            self.INIT_LOGGED_NP_DATA = True

        float_row_data = np.squeeze(float_row_data)
        assert isinstance(float_row_data, np.ndarray)
        assert float_row_data.ndim == 1 or float_row_data.ndim == 0
        if not (key in self.LOGGED_NP_DATA):
            self.LOGGED_NP_DATA[key] = np.zeros((0,float_row_data.size+LOGGED_NP_DATA_HEADER_SIZE))
        assert float_row_data.size + LOGGED_NP_DATA_HEADER_SIZE == self.LOGGED_NP_DATA[key].shape[1]
        s_t = np.array([g_step,self.N_STEP,time()])
        data = np.hstack((s_t,float_row_data))
        self.LOGGED_NP_DATA[key] = np.vstack((self.LOGGED_NP_DATA[key],data))

    def status(self):
        if self.DEBUG:
            pprint.pprint(vars(self))

    def _print(self, *args, **kwargs):
        if self.DEBUG and not StatusObject.DISABLE_ALL_DEBUG and (
                self.N_STEP % self.DEBUG_STEP == 0 or self.N_STEP % self.DEBUG_STEP == 1 or self.N_STEP % self.DEBUG_STEP == 2):
            print(("%6d\t" % self.N_STEP) + " ".join(map(str, args)), **kwargs)

    def _printa(self, *args, **kwargs):
        if self.DEBUG and not StatusObject.DISABLE_ALL_DEBUG:
            print(("%6d\t" % self.N_STEP) + ("%10s\t" % self.__class__.__name__) + " ".join(map(str, args)), **kwargs)

    def _moving_average(self, key, new_value):
        if not self.INIT_MOVING_AVERAGE:
            self.MOVING_AVERAGE_DICT = {}
            self.MOVING_AVERAGE_DICT_N_STEP = {}
            self.INIT_MOVING_AVERAGE = True

        if not (key in self.MOVING_AVERAGE_DICT):
            self.MOVING_AVERAGE_DICT[key] = 0.
            self.MOVING_AVERAGE_DICT_N_STEP[key] = 0.

        if key in self.MOVING_AVERAGE_DICT and key in self.MOVING_AVERAGE_DICT_N_STEP:
            step = self.MOVING_AVERAGE_DICT_N_STEP[key] + 1
            step = step if step < self.MOVING_AVERAGE_TIME_WINDOW else self.MOVING_AVERAGE_TIME_WINDOW

            self.MOVING_AVERAGE_DICT[key] = self.MOVING_AVERAGE_DICT[key] * (1.-1./step) + 1./step * new_value
            self.MOVING_AVERAGE_DICT_N_STEP[key] += 1

            return self.MOVING_AVERAGE_DICT[key]
        else:
            return 0.

    def _debug(self, ENABLE ,debug_step=100):
        self.DEBUG = ENABLE
        self.DEBUG_STEP = debug_step


class NP_LOGGER:
    def __init__(self,path,name):
        self.path = path
        self.name = name

class TIMED_OBJECT:
    def __init__(self):
        self.timers = []
        self.ntimer = 0
    def _get_tic(self):
        self.ntimer += 1
        self.timers.append((self.ntimer,time()))
        return self.ntimer
    def _get_tim(self,tic_id):
        for t in self.timers:
            if t[0] == tic_id:
                tim = t[0]
                self.timers.remove(t)
                return (time()-tim)*1e6
        raise Exception("no timer is found.")


def GET_LOG_PATH_FOR_SIM_SCRIPT(sim_script_path):
    OUT_ALL_SIM_FOLDER = os.path.splitext(os.path.basename(sim_script_path))[0]
    OUT_ALL_SIM_FOLDER = os.path.join(os.path.dirname(os.path.realpath(sim_script_path)), OUT_ALL_SIM_FOLDER)
    try:
        os.mkdir(OUT_ALL_SIM_FOLDER)
    except:
        pass
    SIM_NAME_TIME = os.path.splitext(os.path.basename(sim_script_path))[0] + "-" + get_current_time_str() + "-ail"
    OUT_PER_SIM_FOLDER = os.path.join(OUT_ALL_SIM_FOLDER, SIM_NAME_TIME)
    return OUT_PER_SIM_FOLDER

def GET_FILE_NAME_FOR_SIM_SCRIPT(file):
    FILE_NAME = os.path.splitext(os.path.basename(file))[0]
    return FILE_NAME

def plot_a_array(arr, mavg_n = 20, name= "", script_file = None, postfix = "", idx = None , show = False, save_path = None):
    fig = plt.figure(figsize=(16, 6), dpi=80)
    data = np.convolve(arr, np.ones(mavg_n)/mavg_n, mode='valid')
    if idx:
        plt.plot(idx[:data.size],data)
    else:
        plt.plot(np.arange(start=1, stop=data.size+1),data)

    if show:
        plt.show()
    if save_path:
        if script_file:
            FIG_NAME = "-".join([GET_FILE_NAME_FOR_SIM_SCRIPT(script_file),name,postfix,get_current_time_str()])
        else:
            FIG_NAME = "-".join([name,postfix,get_current_time_str()])
        try:
            os.mkdir(os.path.join(save_path,"saved_figures"))
        except:
            pass
        fig.savefig(os.path.join(save_path,"saved_figures",FIG_NAME))



