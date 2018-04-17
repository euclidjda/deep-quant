#! /usr/bin/env python3

#
# usage: python3 calc-forecast-errors.py forecasts1.txt [forecasts2.txt]
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import sys
import copy
import subprocess

import math
import numpy as np
import regex as re
import pandas as pd
import argparse as ap
import random as random
import time
import glob
import calendar

_EFF_DATE_IDX = 0
_GVKEY_IDX    = 1
_FPE_DATE_IDX = 2
_FORECAST_IDX = 3
_ACTUAL_IDX   = 4
_MIN_VALUE    = 0.01

def read_data(filename):
    with open(filename) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    data = list()
    for i in range(len(content)):
        data.append(content[i])
    return data

def make_key(val1,val2):
    return "%s:%s"%(val1,val2)

def unmake_key(key):
    return key.split(':')

def fill_data_from_file(dict,file,gvkey_idx,date_idx):
    data = read_data(file)
    for line in data:
        fields = line.split(' ')
        #if len(fields) < 5:
        #    print('** '+line+' **')
        #    exit()
        gvkey = fields[gvkey_idx]
        date  = fields[date_idx]
        key = make_key(gvkey,date)
        dict[key] = line

def get_fields(line):
    return line.split(' ')

def main():

    assert(len(sys.argv)>1)
    filename = sys.argv[1]

    forecasts = dict()
    errors = dict()
    
    fill_data_from_file(forecasts,filename,_GVKEY_IDX,_EFF_DATE_IDX)

    # forecast_date gvkey forecast_period_end forecast_mean forecast_med actual_ibcom actual_shares
    count = 0
    for key in sorted(forecasts):
        (gvkey,date) = unmake_key(key)
        data = get_fields(forecasts[key])
        fcst = data[_FORECAST_IDX]
        actual = data[_ACTUAL_IDX]
        # print("%s %s"%(fcst,actual))
        # exit()
        if fcst == 'NULL' or actual == 'NULL':
            continue
        fcst = float(fcst)
        actual = float(actual)
        err = abs(actual-fcst)/abs(max(fcst,_MIN_VALUE))
        if err > 1.0: err = 1.0
        if date not in errors:
            errors[date] = list()
        errors[date].append(err)
        
    for date in sorted(errors):
        errs = errors[date]
        MAE = np.mean(np.array(errs))
        print("%s %.4f"%(date,MAE))
        
if __name__ == "__main__":
    main()
