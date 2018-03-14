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

import math
import numpy as np
import regex as re
import pandas as pd
import argparse as ap
import random as random
import time

_GENERATIONS   = 100
_POP_SIZE      = 40
_NUM_SURVIVORS = 20
_NUM_GPU       = 4
_SLEEP_TIME    = 1
_MUTATE_RATE   = 0.4
_SHELL         = '/bin/sh'
_VALID_ERR_IDX = 7

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

def donefile_filename(gen,gpu):
    return "output/done-g%04d-u%03d.txt"%(gen,gpu)

def script_filename(gen,gpu):
    dirname = 'scripts'
    basename = dirname + "/train-g%04d"%gen
    scriptname = basename + "-u%03d.sh"%gpu
    return scriptname

def serialize_member(mem):
  str = ""
  for el in sorted(mem):
    if el != '--name' and el != '--model_dir':
      str += ':' + mem[el][0]
  return str

def generate_results_test(pop,gen):
  result = list()
  for i in range(len(pop)):
    str = serialize_member(pop[i])
    seed = hash(str)
    random.seed(seed)
    result.append(random.random())
  return result

def generate_results(pop,gen):
    result = list()
    for i in range(len(pop)):
        filename = output_filename(gen,i)
        print("Reading file "+filename)
        with open(filename) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        # remove lines w/o error
        content = [s for s in content if re.search('MSE',s)]
        errors = [float(s.split()[_VALID_ERR_IDX]) for s in content]
        errors.sort()
        result.append(errors[0])
    print("o"*80)
    print(result)
    assert(len(pop) == len(result))
    return result

def poll_for_done(pop,gen):
    not_done = True
    while(not_done):
        time.sleep(_SLEEP_TIME) #
        num_done = 0
        for gpu in range(_NUM_GPU):
            if os.path.isfile(donefile_filename(gen,gpu)):
                num_done += 1
        if num_done == _NUM_GPU:
            not_done = False

def execute_train_scripts(pop,gen):
    str = ""
    for gpu in range(_NUM_GPU):
        str += script_filename(gen,gpu) + " & "
    os.system(str)

def create_train_scripts(pop,gen):
    dirname = 'scripts'
    if os.path.isdir(dirname) is False:
        os.makedirs(dirname)
    if os.path.isdir('output') is False:
        os.makedirs('output')
    if os.path.isdir('chkpts') is False:
        os.makedirs('chkpts')
    for gpu in range(_NUM_GPU):
        scriptname = script_filename(gen,gpu)
        with open(scriptname,"w") as f:
            print("#!%s"%_SHELL,file=f)
            assert(len(pop)%_NUM_GPU==0)
            m = len(pop)//_NUM_GPU
            pop_idxs = [gpu*m + i for i in range(m)]
            for i in pop_idxs:
                str = "CUDA_VISIBLE_DEVICES=%d"%gpu
                str += " deep_quant.py"
                str += " --config=config/"+config_filename(gen,i)
                str += " > output/stdout-%s.txt"%get_name(gen,i)
                str += " 2> output/stderr-%s.txt"%get_name(gen,i)
                str += "; rm -rf chkpts/chkpts-%s"%get_name(gen,i)+";"
                print(str,file=f)
            donefile = donefile_filename(gen,gpu)
            print("echo 'done.' > %s"%donefile,file=f)
        f.closed
        os.system("chmod +x %s"%scriptname)

def write_population_configs(pop,gen):
    dirname = 'config'
    if os.path.isdir(dirname) is not True:
        os.makedirs(dirname)
    for i in range(len(pop)):
        filename = dirname + '/' + config_filename(gen,i)
        configs = pop[i]
        configs['--model_dir'][0] = "chkpts/chkpts-%s"%get_name(gen,i)
        with open(filename,"w") as f:
            for flag in sorted(configs):
                print("%-30s %s"%(flag,configs[flag][0]),file=f)
        f.closed

def train_population(pop,gen):
    """ Train the population
    Args:
      pop is a population
      gen is the generation number (id)
    Returns:
      An array of performance/error for each pop member
    """
    assert(type(pop) is list)
    write_population_configs(pop,gen)
    create_train_scripts(pop,gen)
    #execute_train_scripts(pop,gen)
    #poll_for_done(pop,gen)
    #result = generate_results(pop,gen)
    # result = generate_results_test(pop,gen)
    #return result

def swap(items,i,j):
    """ Swap two items in a list
    """
    assert(type(items) is list)
    tmp = items[i]
    items[i] = items[j]
    items[j] = tmp

def randomize(mem):
    """ Radomize a population memeber
    Args: A member of a pop (dict of lists)
    """
    assert(type(mem) is dict)
    for flag in mem:
        items = mem[flag]
        if len(items) > 1:
            i = random.randrange(0,len(items))
            swap(items,0,i)

def mutate(mem):
    """ Mutate a population memeber
    Args: A member of a pop (dict of lists)
    """
    assert(type(mem) is dict)
    # get flags that have more than one element
    flags = [f for f in mem if len(mem[f]) > 1]
    # randomly choose one
    random.shuffle(flags)
    flag = flags[0]
    # mutate it
    i = random.randrange(1,len(mem[flag]))
    before = mem[flag][0]
    before_s = serialize_member(mem)
    swap(mem[flag],0,i)
    after = mem[flag][0]
    after_s = serialize_member(mem)
    print("mutation: %s: %s -> %s"%(flag,before,after))
    print("BE "+before_s)
    print("AF "+after_s)

def init_population(config):
    """ Initialize a population 
    Args: config
    Returns: population
    """
    pop = list()
    for i in range(_POP_SIZE):
      mem = copy.deepcopy(config)
      randomize(mem)
      mem['--name'] = list()
      mem['--name'].append(get_name(1,i))
      str = serialize_member(mem)
      print("IN %s %s"%(str,hash(str)))
      pop.append(mem)
    return pop

def cross_parents(mom,dad,child_name='none'):
    assert(type(mom) is dict)
    assert(type(dad) is dict)
    child = dict()
    for flag in mom:
      assert(type(mom[flag]) is list)
      assert(type(dad[flag]) is list)
      items = mom[flag] if random.random() > 0.5 else dad[flag]
      child[flag] = items[:] # ensure a copy
    child['--name'][0] = child_name
    print("Crossing (1) x (2) = (3)")
    print("1: " + serialize_member(mom))
    print("2: " + serialize_member(dad))
    print("3: " + serialize_member(child))
    return child

def get_next_generation(pop, gen, results):
    assert(type(pop) is list)
    assert(type(results) is list)
    assert(len(pop) == len(results))
    combined = list(zip(results,pop))
    # lowest values are at top of list
    print('-'*80)
    print(type(combined))
    print(type(combined[0][0]))
    print(type(combined[0][1]))
    combined.sort(key=lambda tup: tup[0])
    new_best = combined[0]
    survivors = [combined[i][1] for i in range(_NUM_SURVIVORS)]
    new_pop = list()
    for i in range(_POP_SIZE):
      # cross two suvivors
      mom = survivors[random.randrange(_NUM_SURVIVORS)]
      dad = survivors[random.randrange(_NUM_SURVIVORS)]
      child = cross_parents(mom,dad,child_name=get_name(gen+1,i))
      # mutations
      if random.random() <= _MUTATE_RATE:
        mutate(child)
      print("FI "+serialize_member(child))
      new_pop.append(child)
    return new_pop, new_best
  
def parse_config(filename):
    with open(filename) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    config = dict()
    for i in range(len(content)):
        elements = content[i].split()
        flag = elements.pop(0)
        config[flag] = elements
    return config

def get_args():
    # read and populate configuration
    parser = ap.ArgumentParser(description='Hyper Parameter Search')
    parser.add_argument('--template', help='Configuration file', required=True)
    parser.add_argument('--genetic', dest='genetic', action='store_true', help='Use Genetic Algo')
    parser.set_defaults(genetic=False)
    args = vars(parser.parse_args())
    return args

def execute_genetic_search(args):
    config_filename = args['template']
    # config is a dict of lists
    config = parse_config(config_filename)
    
    random.seed(config['--seed'][0])
    
    print("Seaching on the following configs:")
    for flag in config:
        if (len(config[flag]) > 1):
            print("  %s -> (%s)"%(flag,','.join(config[flag])))

    results = [float('inf')]*_POP_SIZE

    pop = init_population(config)
    best = None
    for i in range(_GENERATIONS):
        gen = i+1
        result = train_population(pop,gen)
        print('*'*80)
        print(result)
        (pop,new_best) = get_next_generation(pop,gen,result)
        if best is None or best[0] > new_best[0]:
            best = new_best
        name = best[1]['--name'][0]
        error = best[0]
        print("Generation: %s Best: %s Error: %s"%(gen,name,error))

def get_all_config_permutations(src,tbl,i,result):
    flags = [f for f in sorted(src)]
    if i == len(flags):
        result.append(tbl)
    else:
        flag = flags[i]
        curr = src[flag]
        for param in curr:
            new_tbl = tbl.copy()
            new_tbl[flag] = [param]
            get_all_config_permutations(src,new_tbl,i+1,result)
        
def execute_grid_search(args):
    config_filename = args['template']
    # config is a dict of lists
    config = parse_config(config_filename)
    result = list()
    tbl = dict()
    get_all_config_permutations(config,tbl,0,result)
    train_population(result,0)

def main():
    args = get_args()

    if args['genetic'] is True:
        execute_genetic_search(args)
    else:
        execute_grid_search(args)
        
if __name__ == "__main__":
    main()
