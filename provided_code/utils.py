import os
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def load_file(file_path: Path) -> Union[NDArray, Dict[str, NDArray]]:
    """
    Load a file in one of the formats provided in the OpenKBP dataset
    """
    if file_path.stem == "voxel_dimensions":
        return np.loadtxt(file_path)

    loaded_file_df = pd.read_csv(file_path, index_col=0)
    if loaded_file_df.isnull().values.any():  # Data is a mask
        loaded_file = np.array(loaded_file_df.index).squeeze()
    else:  # Data is a sparse matrix
        loaded_file = {"indices": loaded_file_df.index.values, "data": loaded_file_df.data.values}

    return loaded_file


def get_paths(directory_path: Path, extension: Optional[str] = None) -> list[Path]:
    """
    Get the paths of every file contained in `directory_path` that also has the extension `extension` if one is provided.
    """
    all_paths = []

    if not directory_path.is_dir():
        pass
    elif extension is None:
        dir_list = os.listdir(directory_path)
        for name in dir_list:
            if "." != name[0]:  # Ignore hidden files
                all_paths.append(directory_path / str(name))
    else:
        data_root = Path(directory_path)
        for file_path in data_root.glob("*.{}".format(extension)):
            file_path = Path(file_path)
            if "." != file_path.stem[0]:
                all_paths.append(file_path)

    return all_paths


def sparse_vector_function(x, indices=None) -> dict[str, NDArray]:
    """Convert a tensor into a dictionary of the non-zero values and their corresponding indices
    :param x: the tensor or, if indices is not None, the values that belong at each index
    :param indices: the raveled indices of the tensor
    :return:  sparse vector in the form of a dictionary
    """
    if indices is None:
        y = {"data": x[x > 0], "indices": np.nonzero(x.flatten())[-1]}
    else:
        y = {"data": x[x > 0], "indices": indices[x > 0]}
    return y
