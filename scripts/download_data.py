#! /usr/bin/env python3

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
import gzip
import wget

import requests

from six.moves import urllib

data_url   = 'http://data.euclidean.com/open-data/'
data_dir   = 'datasets'
s3_bucket  = 'deep-quant-data'

# we should read this list from file in datasets
remote_files = ['open-dataset.dat.gz']

def maybe_download(directory, filename, url):
  """Download filename from url unless it's already in directory."""
  if not os.path.exists(directory):
    print("Creating directory %s" % directory)
    os.mkdir(directory)
  filepath = os.path.join(directory, filename)
  if not os.path.exists(filepath):
    print("Downloading %s" % (url+filename))
    wget.download(url+filename, out=directory)
    statinfo = os.stat(filepath)
    print("\nSuccesfully downloaded", filename, statinfo.st_size, "bytes")
  else:
    print("file %s already exists in %s"% (filename,directory) )
  return filepath

def gunzip_file(gz_path, new_path):
  """Unzips from gz_path into new_path."""
  if not os.path.exists(new_path):
    print("Unpacking %s to %s" % (gz_path, new_path))
    with gzip.open(gz_path, "rb") as gz_file:
      with open(new_path, "wb") as new_file:
        for line in gz_file:
          new_file.write(line)


def download_data():

    local_files  = [ os.path.splitext(remote_files[i])[0]
                         for i in range(len(remote_files)) ]

    print("Downloading data ...")
    for i in range(len(remote_files)):
        maybe_download(data_dir, remote_files[i], data_url)
        gunzip_file(data_dir+'/'+remote_files[i],
                    data_dir+'/'+local_files[i])

def main():
    download_data()

main()
