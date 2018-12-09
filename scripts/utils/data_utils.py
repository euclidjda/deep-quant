"Provides tools for user to load data as specified."

import os
from datetime import datetime

import pandas as pd

from batch_generator import BatchGenerator

DATASETS_PATH = os.path.join(os.environ['DEEP_QUANT_ROOT'], 'datasets')
# TODO: fix so we don't have to assume 'DEEP_QUANT_ROOT' is defined in the
# environment

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


def build_and_write_trimmed_datfile(open_dataset_path, ticlist_path, 
                                    trimmed_datfile_path):
    """
    Creates a .dat file by trimming open-dataset.dat so that it contains only
    tickers specified by ticlist file at`ticlist_path`. Writes produced .dat
    file at `trimmed_datfile_path`.

    Args:
      open_dataset_path: the path to open-dataset.dat
      ticlist_path: the path to the list of tickers that the produced .dat file
                    should contain.
      trimmed_datfile_path: path where you'd like the trimmed datfile to be
                            placed.
    """
    def get_gvkeys_from_ticlist(ticlist_path):  #TODO: use actual gvkeys
        """
        Returns 'gvkeys' from ticlist.dat as a sorted list.

        NOTE: Right now, 'gvkeys' are not the actual gvkeys that you'd see in
        Compustat. Instead, they're unique identifiers constructed by
        concatenating a numeric id for the exchange (1 for Nasdaq, 2 for NYSE)
        with the ticker name.
        """
        if os.path.isfile(ticlist_path):
            ticlist_df = pd.read_csv(ticlist_path, sep=' ', header=None)
            gvkeys = list()
            for line in ticlist_df.values:
                if line[1] == 'Nasdaq':
                    gvkeys.append('1'+line[0])
                elif line[1] == 'NYSE':
                    gvkeys.append('2'+line[0])
                else:
                    gvkeys.append('9'+line[0])  # TODO: is that best way to handle
                                                # unrecognized market?
        else:
            gvkeys = list()
            
        return gvkeys

    def shave_open_dataset(open_dataset_path, ticlist_path, dest):
        """
        Trims wanted data (in terms of tics and features only; the shaving by
        dates is done by BatchGenerator's constructor), stores shaved .dat file
        at dest.

        NOTE: trimming by features not implemented yet, will rely on a
        feat_map.dat file.
        """
        gvkeys = get_gvkeys_from_ticlist(ticlist_path)
        open_df = pd.read_csv(open_dataset_path, sep=' ', dtype={'gvkey': str})
        shaved_df = open_df[open_df.gvkey.isin(gvkeys)]
        shaved_df.to_csv(dest, sep=' ', index=False)
        print("Successfully trimmed {} as specified by {}, wrote to {}.".format(
            open_dataset_path, ticlist_path, dest))

    shave_open_dataset(open_dataset_path, ticlist_path, trimmed_datfile_path)


def load_all_data(config, is_training_only=False):
    """
    Returns all data as a BatchGenerator object.
    """
    data_path = get_data_path(config.data_dir, config.datafile)
    batches = BatchGenerator(data_path, config, is_training_only=is_training_only)
    
    return batches


def load_train_valid_data(config):
    """
    Returns train_data and valid_data, both as BatchGenerator objects.
    """
    batches = load_all_data(config)

    train_data = batches.train_batches(verbose=True)
    valid_data = batches.valid_batches(verbose=True)

    return train_data, valid_data
