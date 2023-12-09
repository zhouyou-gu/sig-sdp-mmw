import math
import os
from datetime import datetime
import pprint
from time import time



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
        assert float_row_data.ndim == 1
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

    def _debug(self, debug_step=100):
        self.DEBUG = True
        self.DEBUG_STEP = debug_step