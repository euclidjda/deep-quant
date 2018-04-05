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

_DATE_IDX  = 0
_GVKEY_IDX = 1
_VALUE_IDX = 8

def read_data(filename):
    with open(filename) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    data = list()
    for i in range(len(content)):
        data.append(content[i])
    return data

def get_file_list(prefix):
    files = glob.glob(prefix+'*')
    return files

def get_preds_from_file(preds,file):
    data = read_data(file)
    for line in data:
        fields = line.split(' ')
        gvkey = fields[_GVKEY_IDX]
        date  = fields[_DATE_IDX]
        key = "%s-%s"%(gvkey,date)
        value = float(fields[_VALUE_IDX])
        if key not in preds:
            preds[key] = list()
        preds[key].append(value)


def main():
    assert(len(sys.argv)>1)
    dir_name = sys.argv[1]
    dirs = get_file_list(dir_name)
    dirs.sort()

    preds = dict()

    for dir in dirs:
        # print("Reading files in: "+dir)
        files = get_file_list(dir+'/test-preds-')
        files.sort()
        for file in files:
            # print("Reading predictions from "+file)
            get_preds_from_file(preds,file)

    # write predictions
    count = 0
    for key in sorted(preds):
        (gvkey,date) = key.split('-')
        values = preds[key]
        avg = np.mean(np.array(values))
        med = np.median(np.array(values))
        print("%s %s %s %s"%(date,gvkey,avg,med))
        count+=1

if __name__ == "__main__":
    main()
