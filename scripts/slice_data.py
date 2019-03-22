#! /usr/bin/env python3

#
# usage: slice_data.py start_date end_data < in-file.dat > out-file.dat


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import sys

def main():
    assert(len(sys.argv)>2)
    start_date = sys.argv[1]
    end_date  = sys.argv[2]

    for line in sys.stdin:
        line = line.strip()
        fields = line.split(' ')
        date = fields[0]
        if (date >= start_date) and (date <= end_date):
            print(line)

if __name__ == "__main__":
    main()
