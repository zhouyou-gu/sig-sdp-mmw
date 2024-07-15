import os
from os.path import expanduser


# add your path here as: default_path = "somepath/im-mmw";
# otherwise, the codes assume the path is "~/im-mmw"
default_path = None


def get_working_dir_path():
    if default_path == None:
        home = expanduser("~")
        path = os.path.join(home,"im-mmw")
        return path
    else:
        return default_path