#!/bin/sh
''''exec python3 -u -- "$0" ${1+"$@"} # '''

# #! /usr/bin/env python3
# Copyright 2016 Euclidean Technologies Management LLC All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import sys
import copy
import subprocess
import os

import math
import numpy as np
import regex as re
import pandas as pd
import argparse as ap
import random as random
import time
import configs as configurations
import pickle

_SHELL         = '/bin/sh'
_VALID_ERR_IDX = 7

def get_configs():
    """
    Defines the configurations for hyper parameter search
    """
    configurations.DEFINE_string("configs_fname",None,"CSV containing all the configs to run")
    configurations.DEFINE_boolean("predict",True,"Run predictions after training")
    configurations.DEFINE_integer("num_threads",4,"NUmber of parallel threads (Number of parallel executions)")
    configurations.DEFINE_integer("num_gpu",1,"Number of GPU on the machine, Use 0 if there are None")
    configurations.DEFINE_integer("sleep_time",1,"Sleep time")
    configurations.DEFINE_integer("start_date",None,"First date for prediction on as YYYYMM")
    configurations.DEFINE_integer("end_date",None,"Last date for prediction on as YYYYMM")

    c = configurations.ConfigValues()

    return c


def read_configs(fname):
    """
    :param fname: CSV file containing the configs
    :return: list of member configs as dicts
    """
    # Read CSV
    df = pd.read_csv(fname)
    df = df.applymap(str)

    # Get field names
    cols = df.columns.tolist()
    cols = ['--'+x for x in cols]
    df.columns = cols

    # Initialize empty pop list
    pop_list = []
    for ix, row in df.iterrows():
        mem_dict = dict(zip(cols, row.values))
        pop_list.append(mem_dict)

    return pop_list


def get_name(gen,i):
    d1 = max(6,len(str(gen)))
    d2 = max(6,len(str(i)))
    fmt = 'gen-%0'+str(d1)+'d-mem-%0'+str(d2)+'d';
    return fmt%(gen,i);


def output_filename(gen,i):
    name = get_name(gen,i)
    filename = "output/stdout-%s.txt"%name
    return filename


def config_filename(gen,i):
    name = get_name(gen,i)
    return "%s.conf"%name


def script_filename(gen,thread):
    print(gen)
    dirname = 'scripts'
    basename = dirname + "/train-g%04d".format(gen)
    scriptname = basename + "-u%03d.sh"%thread
    return scriptname


def execute_train_scripts(gen=0):
    """
    Executes the train scripts corresponding to pop list
    :param gen: default 0
    :return:
    """
    str1 = ""
    for thread in range(_NUM_THREADS):
        str1 += script_filename(gen, thread) + " & "
    os.system(str1)


def create_train_scripts(pop, args, gen=0):
    """
    Creates training scripts for member configs in pop
    :param pop: list containing member dicts
    :param args: config args
    :param gen: Default 0
    :return:
    """
    dirname = 'scripts'
    if os.path.isdir(dirname) is False:
        os.makedirs(dirname)
    if os.path.isdir('output') is False:
        os.makedirs('output')
    if os.path.isdir('chkpts') is False:
        os.makedirs('chkpts')
    for thread in range(_NUM_THREADS):
        scriptname = script_filename(gen,thread)
        with open(scriptname,"w") as f:
            print("#!%s"%_SHELL,file=f)
            assert(len(pop)%_NUM_THREADS==0)
            m = len(pop)//_NUM_THREADS
            pop_idxs = [thread*m + i for i in range(m)]
            for i in pop_idxs:
                id_seed = int(17*gen + i + 1)
                # Add GPU number to the members of the generation
                if _NUM_GPU!=0:
                    str1 = "CUDA_VISIBLE_DEVICES=%d"%(thread%_NUM_GPU)
                elif _NUM_GPU==0:
                    str1 = "CUDA_VISIBLE_DEVICES=''"
                str1 += " /home/lchauhan/deep-quant/scripts/deep_quant.py"
                str1 += " --config=config/"+config_filename(gen, i)
                # str1 += " --seed=%i"%id_seed
                str1 += " --cache_id=" + str(id_seed)
                str1 += " > " + output_filename(gen, i) + "-train"
                str1 += " 2> output/train-stderr-%s.txt"%get_name(gen, i)
                print(str1, file=f)
                del str1

                if args.predict:
                    assert(args.start_date is not None)
                    assert(args.end_date is not None)

                    # Add GPU number to the members of the generation
                    if _NUM_GPU != 0:
                        str1 = "CUDA_VISIBLE_DEVICES=%d" % (thread % _NUM_GPU)
                    elif _NUM_GPU == 0:
                        str1 = "CUDA_VISIBLE_DEVICES=''"
                    str1 += " /home/lchauhan/deep-quant/scripts/deep_quant.py"
                    str1 += " --config=config/" + config_filename(gen, i)
                    # str1 += " --seed=%i"%id_seed
                    str1 += " --cache_id=" + str(99*(id_seed+1)) # diff seed for prediction
                    str1 += " --train=False"
                    str1 += " --start_date=%i"%args.start_date
                    str1 += " --end_date=%i"%args.end_date
                    str1 += " --mse_outfile=" + output_filename(gen, i) + "-mse"
                    str1 += " > " + output_filename(gen, i) + "-pred"
                    str1 += " 2> output/pred-stderr-%s.txt" % get_name(gen, i)
                    print(str1, file=f)
                    del str1

        f.closed
        os.system("chmod +x %s"%scriptname)


def write_population_configs(pop):
    """
    Writes the member config files
    :param pop: list of member dicts
    :return:
    """
    dirname = 'config'
    gen = 0
    if os.path.isdir(dirname) is not True:
        os.makedirs(dirname)
    for i in range(len(pop)):
        filename = dirname + '/' + config_filename(gen, i)
        configs = pop[i]
        configs['--model_dir'] = "chkpts/chkpts-%s"%get_name(gen, i)
        with open(filename, "w") as f:
            for flag in sorted(configs):
                print("%-30s %s"%(flag, configs[flag]), file=f)
        f.closed
    return


def train_population(pop,args):
    """
    Runs the training for members in pop
    :param pop: population of config dicts
    :param args: config arguments
    :return:
    """
    assert(type(pop) is list)
    write_population_configs(pop)
    create_train_scripts(pop, args)
    execute_train_scripts(pop)
    return


def execute_training(args):

    assert(args.configs_fname is not None)
    # Read user specified configs population
    pop = read_configs(args.configs_fname)

    train_population(pop, args)

    sys.stdout.flush()


def main():

    config_args = get_configs()

    # Define Global Variables
    global _NUM_THREADS
    global _NUM_GPU
    global _SLEEP_TIME

    _NUM_THREADS   = config_args.num_threads
    _NUM_GPU       = config_args.num_gpu
    _SLEEP_TIME    = config_args.sleep_time

    execute_training(config_args)


if __name__ == "__main__":
    main()
