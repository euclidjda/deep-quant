"Provides tools for user to load data as specified."

import os

import pandas as pd

from batch_generator import BatchGenerator


def get_data_path(data_dir, filename):
    """
    Construct the data path for the experiement. If DEEP_QUANT_ROOT is
    defined in the environment, then the data path is relative to it.

    Args:
      data_dir: the directory name where experimental data is held
      filename: the data file name
    Returns:
      If DEEP_QUANT_ROOT is defined, the fully qualified data path is returned
      Otherwise a path relative to the working directory is returned
    """
    path = os.path.join(data_dir, filename)
    if data_dir != '.' and 'DEEP_QUANT_ROOT' in os.environ:
        path = os.path.join(os.environ['DEEP_QUANT_ROOT'], path)
    return path

def load_all_data(config):
    """
    Returns all data as a BatchGenerator object.
    """
    if config.datasource == "big_datafile":
        all_data_path = get_data_path(config.data_dir, config.datafile)
        batches = BatchGenerator(all_data_path, config)
    elif config.datasource == "WRDS":
        raise Exception("Not Implemented yet, sorry!")
    else:
        raise Exception("Unknown datasource.")  # TODO: use argparse to check

    return batches

def load_train_valid_data(config):
    """
    Returns train_data and valid_data, both as BatchGenerator objects.
    """
    batches = load_all_data(config)

    train_data = batches.train_batches()
    valid_data = batches.valid_batches()

    return train_data, valid_data
