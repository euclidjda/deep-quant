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

def get_search_configs():
    """
    Defines the configurations for hyper parameter search
    """
    configurations.DEFINE_string("template",None,"Template file for hyper-param search")
    configurations.DEFINE_string("search_algorithm","genetic","Algorithm for hyper-param optimization. Select from 'genetic', 'grid_search'")
    configurations.DEFINE_integer("generations",100,"Number of generations for genetic algorithm")
    configurations.DEFINE_integer("pop_size",20,"Population size for genetic algorithm")
    configurations.DEFINE_integer("num_survivors",10,"Number of survivors for genetic algorithm")
    configurations.DEFINE_integer("num_threads",4,"NUmber of parallel threads (Number of parallel executions)")
    configurations.DEFINE_integer("num_gpu",1,"Number of GPU on the machine, Use 0 if there are None")
    configurations.DEFINE_integer("sleep_time",1,"Sleep time")
    configurations.DEFINE_float("mutate_rate",0.02,"Mutation rate for genetic algorithm")
    configurations.DEFINE_string("init_pop",None,"Specify starting population. Path to the pickle file")

    c = configurations.ConfigValues()

    return c

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

def donefile_filename(gen,thread):
    return "output/done-g%04d-u%03d.txt"%(gen,thread)

def script_filename(gen,thread):
    dirname = 'scripts'
    basename = dirname + "/train-g%04d"%gen
    scriptname = basename + "-u%03d.sh"%thread
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
        if len(errors) > 0:
            errors.sort()
            result.append(errors[0])
        else:
            result.append(float('inf'))
        if result[-1] == 'nan':
            result[-1] = float('inf')

    print("-"*80)
    print(result)
    assert(len(pop) == len(result))
    return result

def poll_for_done(pop,gen):
    not_done = True
    while(not_done):
        time.sleep(_SLEEP_TIME) #
        num_done = 0
        for thread in range(_NUM_THREADS):
            if os.path.isfile(donefile_filename(gen,thread)):
                num_done += 1
        if num_done == _NUM_THREADS:
            not_done = False

def execute_train_scripts(pop,gen):
    str = ""
    for thread in range(_NUM_THREADS):
        str += script_filename(gen,thread) + " & "
    os.system(str)

def create_train_scripts(pop,gen):
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
                id_seed = int(17*gen + i)
                # Add GPU number to the members of the generation
                if _NUM_GPU!=0:
                    str = "CUDA_VISIBLE_DEVICES=%d"%(thread%_NUM_GPU)
                elif _NUM_GPU==0:
                    str = "CUDA_VISIBLE_DEVICES=''"
                str += " deep_quant.py"
                str += " --config=config/"+config_filename(gen,i)
                #str += " --seed=%i"%id_seed
                str += " --cache_id=%i"%id_seed
                str += " > " + output_filename(gen,i) 
                str += " 2> output/stderr-%s.txt"%get_name(gen,i)
                #str += "; rm -rf chkpts/chkpts-%s"%get_name(gen,i)+";"
                print(str,file=f)
            donefile = donefile_filename(gen,thread)
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
    execute_train_scripts(pop,gen)
    poll_for_done(pop,gen)
    result = generate_results(pop,gen)
    #result = generate_results_test(pop,gen)
    return result

def calc_diversity(pop):
    mems = [serialize_member(m) for m in pop]
    count = float(len(mems))
    uniq  = float(len(set(mems)))
    assert(count > 0)
    return uniq/count

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
    #print(type(combined))
    #print(type(combined[0][0]))
    #print(type(combined[0][1]))
    combined.sort(key=lambda tup: tup[0])
    new_best = combined[0]
    survivors = [combined[i][1] for i in range(_NUM_SURVIVORS)]
    new_pop = list()
    for i in range(_POP_SIZE):
      # cross two suvivors
      random.shuffle(survivors)
      mom = survivors[0]
      dad = survivors[1]
      child = cross_parents(mom,dad,child_name=get_name(gen+1,i))
      # mutations
      if random.random() <= _MUTATE_RATE:
        mutate(child)
      new_pop.append(child)
    return new_pop, new_best

def parse_config(filename):
    with open(filename) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    # and remove empty lines
    content = [x.strip() for x in content if len(x)]
    config = dict()
    for i in range(len(content)):
        elements = content[i].split()
        flag = elements.pop(0)
        config[flag] = elements
    return config

def execute_genetic_search(args):
    config_filename = args.template
    # config is a dict of lists
    config = parse_config(config_filename)

    random.seed(config['--seed'][0])

    print("Seaching on the following configs:")
    for flag in config:
        if (len(config[flag]) > 1):
            print("  %s -> (%s)"%(flag,','.join(config[flag])))

    results = [float('inf')]*_POP_SIZE

    # Read user specified or latest population
    if args.init_pop:
        pop = pickle.load(open(str(args.init_pop),"rb"))
    else:
        pop = init_population(config)

    best = None
    for i in range(_GENERATIONS):
        gen = i+1

        # Save the latest generation
        dir = "_latest_pop"
        if not os.path.exists(dir):
            os.makedirs(dir)
        pickle.dump(pop,open("_latest_pop/latest_pop.pkl","wb"))

        result = train_population(pop,gen)
        diversity = calc_diversity(pop)
        (pop,new_best) = get_next_generation(pop,gen,result)
        if best is None or best[0] > new_best[0]:
            best = new_best
        best_name = best[1]['--name'][0]
        error = float(best[0])
        print("Generation: %s Best: %s Error: %.4f Diversity: %3d%%"%(gen,best_name,error,int(100*diversity)))
        sys.stdout.flush()

def get_all_config_permutations(src,tbl,i,allperms):
    flags = [f for f in sorted(src)]
    if i == len(flags):
        allperms.append(tbl)
    else:
        flag = flags[i]
        curr = src[flag]
        for param in curr:
            new_tbl = tbl.copy()
            new_tbl[flag] = [param]
            get_all_config_permutations(src,new_tbl,i+1,allperms)

def execute_grid_search(args):
    config_filename = args.template
    # config is a dict of lists
    config = parse_config(config_filename)
    allperms = list()
    tbl = dict()
    get_all_config_permutations(config,tbl,0,allperms)
    train_population(allperms,0)

def main():

    config_search_args = get_search_configs()

    # Define Global Variables
    global _GENERATIONS
    global _POP_SIZE
    global _NUM_SURVIVORS
    global _NUM_THREADS
    global _NUM_GPU
    global _SLEEP_TIME
    global _MUTATE_RATE

    _GENERATIONS   = config_search_args.generations
    _POP_SIZE      = config_search_args.pop_size
    _NUM_SURVIVORS = config_search_args.num_survivors
    _NUM_THREADS   = config_search_args.num_threads
    _NUM_GPU       = config_search_args.num_gpu
    _SLEEP_TIME    = config_search_args.sleep_time
    _MUTATE_RATE   = config_search_args.mutate_rate

    if config_search_args.search_algorithm == 'genetic':
        execute_genetic_search(config_search_args)
    elif config_search_args.search_algorithm == 'grid_search':
        execute_grid_search(config_search_args)
    else:
        print("No search algorithm specified. Selecting default = genetic")
        execute_genetic_search(config_search_args)

if __name__ == "__main__":
    main()
