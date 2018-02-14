"Provides tools for user to load data as specified."

import os
import csv
from datetime import datetime

import pandas as pd

from batch_generator import BatchGenerator

BIG_DF_PATH = os.path.join(os.environ['DEEP_QUANT_ROOT'], 'data', 'datasets', 
                           'big_datafile.dat')  # TODO: fix so we don't have to
                                                # assume 'DEEP_QUANT_ROOT' is
                                                # defined in the environment

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


def get_gvkeys_from_tkrlist(tkrlist):  #TODO: use actual gvkeys
    """
    Returns 'gvkeys' from tkrlist.csv as a sorted list.

    NOTE: Right now, 'gvkeys' are not the actual gvkeys that you'd see in
    Compustat. Instead, they're unique identifiers constructed by concatenating
    a numeric id for the exchange (1 for Nasdaq, 2 for NYSE) with the ticker
    name.
    """
    tkrlist_filepath = os.path.join(os.environ['DEEP_QUANT_ROOT'], 'data',
                                    'tkrlists', "{}.csv".format(tkrlist))

    if os.path.isfile(tkrlist_filepath):
        with open(tkrlist_filepath, 'r') as f:
            reader = csv.reader(f)
            lines = list(reader)

        gvkeys = list()
        for line in lines:
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


def shave_big_datafile(config):
    """
    Shaves wanted data (in terms of tkrs and features only; the shaving by dates
    is done by BatchGenerator's constructor), returns path to shaved .dat file.

    NOTE: shaving by features not implemented yet, will rely on a feat_map.txt
    file.
    """
    gvkeys = get_gvkeys_from_tkrlist(config.tkrlist)
    big_df = pd.read_csv(BIG_DF_PATH, sep=' ', dtype={config.key_field: str})
    shaved_df = big_df[big_df.gvkey.isin(gvkeys)]
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    shaved_data_path = os.path.join(os.path.split(BIG_DF_PATH)[0],
                                    "shaved_{}.dat".format(now))
    shaved_df.to_csv(shaved_data_path, sep=' ', index=False)
    return shaved_data_path


def write_WRDS_data(config):
    """
    Writes .dat file using data from WRDS.
    """
    raise NotImplementedError("Sorry! WRDS integration not ready.")  # TODO


def load_wanted_data(config):
    """
    Returns all data as a BatchGenerator object.
    """
    if config.datasource == "big_datafile":
        data_path = shave_big_datafile(config)
    elif config.datasource == "WRDS":
        data_path = write_WRDS_data(config)
    else:
        raise Exception("Unknown datasource.")  # TODO: use argparse to check
    
    batches = BatchGenerator(data_path, config)
    
    #cleanup
    if config.datasource == "big_datafile":
        os.remove(data_path)

    return batches


def load_train_valid_data(config):
    """
    Returns train_data and valid_data, both as BatchGenerator objects.
    """
    batches = load_wanted_data(config)

    train_data = batches.train_batches()
    valid_data = batches.valid_batches()

    return train_data, valid_data
