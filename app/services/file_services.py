import ntpath
import os
import shutil
from os import listdir
from os.path import isfile, join
from shutil import copyfile


def get_folders(my_path, ext=None):
    dir_paths = [d for d in listdir(my_path) if not isfile(join(my_path, d))]

    result = []
    for fp in dir_paths:
        full_path = os.path.join(my_path, fp)
        if ext is not None:
            if fp.endswith(ext):
                result.append(full_path)
        else:
            result.append(full_path)

    return result


def get_files(my_path, ext=None):
    file_paths = [f for f in listdir(my_path) if isfile(join(my_path, f))]

    result = []
    for fp in file_paths:
        full_path = os.path.join(my_path, fp)
        if ext is not None:
            if fp.endswith(ext):
                result.append(full_path)
        else:
            result.append(full_path)

    return result


def walk_dir(dir_path, ext=None):
    result = []
    for r, d, f in os.walk(dir_path):
        for file in f:
            if ext is not None and file.endswith(ext):
                result.append(os.path.join(r, file))

    return result


def copy_file(file_paths, destination):
    for f in file_paths:
        dest_path = os.path.join(destination, ntpath.basename(f))
        copyfile(f, dest_path)


def empty_dir(folder_path):
    shutil.rmtree(folder_path)
    os.makedirs(folder_path)
