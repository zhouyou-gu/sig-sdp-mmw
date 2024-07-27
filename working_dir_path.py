from os.path import dirname, abspath

def get_working_dir_path():
    return dirname(dirname(abspath(__file__)))