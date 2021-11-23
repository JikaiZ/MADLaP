################################################
# Helper functions
################################################
import os
import time
import pandas as pd
import numpy as np

def get_mrn_list(folder_name, filename):
    """given a folder name and a filename, read the list"""
    with open(os.path.join(folder_name, filename), 'r') as f:
        content = f.readlines()
    selected_mrns = [x.strip() for x in content]
    return selected_mrns

def return_folder_name(file_path):
    """Return a list of folder names, usually MRNs, for a given path"""
    name_list = []
    for path_name in os.listdir(file_path):
        subpath_name = os.path.join(file_path, path_name)
        if os.path.isdir(subpath_name):
            name_list.append(path_name)
    name_list = np.array(name_list)

    return name_list


def write_file(x, destination):
    """a helper function that writes the input to test.txt for displaying results"""
    if isinstance(x, list):
        with open(destination, 'w') as f:
            for item in x:
                f.write('%s\n' % item)
    if isinstance(x, dict):
        with open(destination, 'w') as f:
            f.write(json.dumps(x))