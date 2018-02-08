#!/bin/sh
''''exec python3 -u -- "$0" ${1+"$@"} # '''

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

import numpy as np
import regex as re
import pandas as pd
import argparse as ap

def train_population():
  pass

def create_population():
  pass

def parse_config(filename):
  with open(filename) as f:
    content = f.readlines()
  # you may also want to remove whitespace characters like `\n` at the end of each line
  content = [x.strip() for x in content] 
  config = dict()
  for i in range(len(content)):
    elements = content[i].split()
    flag = elements.pop(0)
    config[flag] = elements
  return config

def get_config_filename():
  # read and populate configuration
  parser = ap.ArgumentParser(description="Hyper Parameter Search")
  parser.add_argument("--config", help="Configuration file", required=True)
  args = vars(parser.parse_args())
  config_filename = args['config']
  return config_filename

def main():
  config_filename = get_config_filename()
  # config is a dict of lists
  config = parse_config(config_filename)

  print("Seaching on the following configs:")
  for flag in config:
    if (len(config[flag]) > 1):
      print("  %s -> (%s)"%(flag,','.join(config[flag])))

  results = list()

  while(1):
    pop = create_population(config,results)
    results = train_population(pop)
    

if __name__ == "__main__":
  main()
