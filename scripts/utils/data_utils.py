"Provides tools for user to load data as specified."

import os
from datetime import datetime

import pandas as pd

from batch_generator_new import BatchGenerator

DATASETS_PATH = os.path.join(os.environ['DEEP_QUANT_ROOT'], 'datasets')
OPEN_DF_PATH = os.path.join(DATASETS_PATH, 'open_dataset.dat') 
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


def create_datafile(datasource, ticlist, dest_basename):
    """
    Creates a .dat file using `datasource` (either `open_dataset` or `WRDS`).
    This file will be made up of tickers from ticlist, and will reside in
    `DEEP_QUANT_ROOT`/datasets/dest_basename.

    Args:
      datasource: specifies whether the datafile should be built using
                  open_dataset as a starting point, or be built by querying
                  WRDS.
      ticlist: the name of the list of tickers that the produced .dat file
               should contain. NOTE: this should be postfixed by .dat.
      dest_basename: the name you'd like the produced datafile to have. NOTE:
                     this should be postfixed by .dat.
    """
    def get_gvkeys_from_ticlist(ticlist):  #TODO: use actual gvkeys
        """
        Returns 'gvkeys' from ticlist.dat as a sorted list.

        NOTE: Right now, 'gvkeys' are not the actual gvkeys that you'd see in
        Compustat. Instead, they're unique identifiers constructed by concatenating
        a numeric id for the exchange (1 for Nasdaq, 2 for NYSE) with the ticker
        name.
        """
        ticlist_filepath = os.path.join(DATASETS_PATH, ticlist)

        if os.path.isfile(ticlist_filepath):
            ticlist_df = pd.read_csv(ticlist_filepath, sep=' ', header=None)
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

    def shave_open_dataset(ticlist, dest):
        """
        Shaves wanted data (in terms of tics and features only; the shaving by
        dates is done by BatchGenerator's constructor), stores shaved .dat file
        at dest.

        NOTE: shaving by features not implemented yet, will rely on a
        feat_map.txt file.
        """
        gvkeys = get_gvkeys_from_ticlist(ticlist)
        open_df = pd.read_csv(OPEN_DF_PATH, sep=' ', dtype={'gvkey': str})
        shaved_df = open_df[open_df.gvkey.isin(gvkeys)]
        shaved_df.to_csv(dest, sep=' ', index=False)

    def write_WRDS_data(dest):
        """
        Writes .dat file using data from WRDS.
        """
        raise NotImplementedError("Sorry! WRDS integration not ready.")  # TODO

    dest = get_data_path(DATASETS_PATH, dest_basename)

    if datasource == "open_dataset":
        shave_open_dataset(ticlist, dest)
    elif datasource == "WRDS":
        write_WRDS_data(ticlist, dest)
    else:
        raise Exception("Unknown datasource.")


def load_all_data(config):
    """
    Returns all data as a BatchGenerator object.
    """
    data_path = get_data_path(config.data_dir, config.datafile)
    batches = BatchGenerator(data_path, config)
    
    return batches


def load_train_valid_data(config):
    """
    Returns train_data and valid_data, both as BatchGenerator objects.
    """
    batches = load_all_data(config)

    train_data = batches.train_batches()
    valid_data = batches.valid_batches()

    return train_data, valid_data
