import os
import pathlib

import numpy as np
import pandas as pd


def load_file(file_name):
    """Load a file in one of the formats provided in the OpenKBP dataset
    :param file_name: the name of the file to be loaded
    :return: the file loaded
    """
    # Load the file as a csv
    loaded_file_df = pd.read_csv(file_name, index_col=0)
    # If the csv is voxel dimensions read it with numpy
    if 'voxel_dimensions.csv' in file_name:
        loaded_file = np.loadtxt(file_name)
    # Check if the data has any values
    elif loaded_file_df.isnull().values.any():
        # Then the data is a vector, which we assume is for a mask of ones
        loaded_file = np.array(loaded_file_df.index).squeeze()
    else:
        # Then the data is a matrix of indices and data points
        loaded_file = {'indices': np.array(loaded_file_df.index).squeeze(),
                       'data': np.array(loaded_file_df['data']).squeeze()}

    return loaded_file


def get_paths(directory_path, ext=''):
    """Get the paths of every file with a specified extension in a directory
    :param directory_path: the path of the directory of interest
    :param ext: the extensions of the files of interest
    :return: the path of all files of interest
    """
    # if dir_name doesn't exist return an empty array
    if not os.path.isdir(directory_path):
        return []
    # Otherwise dir_name exists and function returns contents name(s)
    else:
        all_image_paths = []
        # If no extension given, then get all files
        if ext == '':
            dir_list = os.listdir(directory_path)
            for iPath in dir_list:
                if '.' != iPath[0]:  # Ignore hidden files
                    all_image_paths.append('{}/{}'.format(directory_path, str(iPath)))
        else:
            # Get list of paths for files with the extension ext
            data_root = pathlib.Path(directory_path)
            for iPath in data_root.glob('*.{}'.format(ext)):
                all_image_paths.append(str(iPath))

    return all_image_paths


def get_paths_from_sub_directories(main_directory_path, sub_dir_list, ext=''):
    """Compiles a list of all paths within each sub directory listed in sub_dir_list that follows the main_dir_path
    :param main_directory_path: the path for the main directory of interest
    :param sub_dir_list: the name(s) of the directory of interest that are in the main_directory
    :param ext: the extension of the files of interest (in the usb directories)
    :return:
    """
    # Initialize list of paths
    path_list = []
    # Iterate through the sub directory names and build up the path list
    for sub_dir in sub_dir_list:
        paths_to_add = get_paths('{}/{}'.format(main_directory_path, sub_dir), ext=ext)
        path_list.extend(paths_to_add)

    return path_list


def sparse_vector_function(x, indices=None):
    """Convert a tensor into a dictionary of the non zero values and their corresponding indices
    :param x: the tensor or, if indices is not None, the values that belong at each index
    :param indices: the raveled indices of the tensor
    :return:  sparse vector in the form of a dictionary
    """
    if indices is None:
        y = {'data': x[x > 0], 'indices': np.nonzero(x.flatten())[-1]}
    else:
        y = {'data': x[x > 0], 'indices': indices[x > 0]}
    return y


def make_directory_and_return_path(dir_path):
    """Makes a directory only if it does not already exist
    :param dir_path: the path of the directory to be made
    :return: returns the directory path
    """
    os.makedirs(dir_path, exist_ok=True)

    return dir_path
