#! /usr/bin/env python3

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
_TIC_IDX      = 2
_FPE_DATE_IDX = 3
_MEAN_IDX     = 2
_MEDIAN_IDX   = 3
_IBCOM_IDX    = 5
_SHARES_IDX   = 6

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

def fill_data_from_file(dict,file,idx1,idx2):
    data = read_data(file)
    for line in data:
        fields = line.split(' ')
        gvkey = fields[idx1]
        date  = fields[idx2]
        key = make_key(gvkey,date)
        dict[key] = line

def get_fields(line):
    return line.split(' ')

def add_one_year(date):
    year  = date[0:4]
    month = date[5:7]
    day   = date[8:]
    year  = int(year)+1
    if (calendar.isleap(year) and month == '02'):
        day = '29'
    return "%s-%s-%s"%(year,month,day)

def main():
    assert(len(sys.argv)>2)
    actuals_filename   = sys.argv[1]
    forecasts_filename = sys.argv[2]

    fpe_date_to_actuals   = dict() # gvkey|YYYY-MM-DD -> actual
    eff_date_to_actuals   = dict() # gvkey|YYYYMM -> actuals
    eff_date_to_forecasts = dict() # gvkey|YYYYMM -> forecasts

    fill_data_from_file(eff_date_to_forecasts,forecasts_filename,_GVKEY_IDX,_EFF_DATE_IDX)
    fill_data_from_file(eff_date_to_actuals,actuals_filename,_GVKEY_IDX,_EFF_DATE_IDX)
    fill_data_from_file(fpe_date_to_actuals,actuals_filename,_GVKEY_IDX,_FPE_DATE_IDX)

    # forecast_date gvkey forecast_period_end forecast_mean forecast_med actual_ibcom actual_shares
    count = 0
    for key in sorted(eff_date_to_forecasts):
        (gvkey,eff_date) = unmake_key(key)
        forecasts = get_fields(eff_date_to_forecasts[key])
        eff_actuals = get_fields(eff_date_to_actuals[key])
        tic = eff_actuals[_TIC_IDX]
        ppe_date  = eff_actuals[_FPE_DATE_IDX]
        fpe_date  = add_one_year(ppe_date)
        fpe_actuals = None
        fpe_key = make_key(gvkey,fpe_date)
        fpe_actuals = get_fields(fpe_date_to_actuals[fpe_key]) if fpe_key in fpe_date_to_actuals else None
        
        count+=1
        # if count > 50: exit()

        # Now we can print the record
        print("%s"%eff_date,end=' ')
        print("%s"%gvkey,end=' ')
        print("%s"%tic,end=' ')
        print("%s"%fpe_date,end=' ')

        print("%s"%(forecasts[_MEAN_IDX]),end=' ')
        print("%s"%(forecasts[_MEDIAN_IDX]),end=' ')

        if fpe_actuals is None:
            print("NULL NULL",end=' ')
        else:
            print("%s"%(fpe_actuals[_IBCOM_IDX]),end=' ')
            print("%s"%(fpe_actuals[_SHARES_IDX]),end=' ')
        print()


if __name__ == "__main__":
    main()
