#! /usr/bin/env python3

#
# usage: python3 scripts/cnsns-csv-to-dat.py cnsns-est-data.csv min_cnt > cnsns-est-data.dat
# min_cnt is the min number of analysts required in an estimate

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

_GVKEY_IDX    = 1
_IID_IDX      = 2
_EFF_DATE_IDX = 3
_FPE_DATE_IDX = 4
_FCST_IDX     = 5
_CNT_IDX      = 6
_ACTUAL_IDX   = 7

def read_data(filename):
    with open(filename) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    data = list()
    for i in range(len(content)):
        data.append(content[i])
    return data

def get_fields(line):
    return line.split(',')

def convert_eff_date(orig):
    year  = orig[0:4]
    month = orig[5:7]
    new   = year+month
    return new

def main():
    assert(len(sys.argv)>2)
    filename = sys.argv[1]
    min_cnt  = sys.argv[2]
    data = read_data(filename)

    for line in data:
        fields = get_fields(line)
        cnt = fields[_CNT_IDX]
        if cnt < min_cnt: continue
        effdate = convert_eff_date(fields[_EFF_DATE_IDX])
        fpedate = fields[_FPE_DATE_IDX]
        key = fields[_GVKEY_IDX] + fields[_IID_IDX]
        fcst = fields[_FCST_IDX]
        actual = fields[_ACTUAL_IDX]
        print("%s %s %s %s %s"%(effdate,key,fpedate,fcst,actual))

if __name__ == "__main__":
    main()
