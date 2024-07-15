import os
import time
from multiprocessing import Process
from os.path import expanduser
from working_dir_path import *

PATH = os.path.dirname(os.path.realpath(__file__))
TEST_LIST = [
    "sim-eta-025.py",
    "sim-eta-050.py",
    "sim-eta-075.py",
    "sim-eta-100.py"
]

CMD_LIST = []
for t in TEST_LIST:
    test_path = os.path.join(PATH,t)
    cmd = "PYTHONPATH=" + get_working_dir_path() + " python3 " + test_path
    os.system(cmd)