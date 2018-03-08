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

import os
import gzip
import wget

import argparse

from utils.data_utils import build_and_write_trimmed_datfile

data_url = 'http://data.euclidean.com/open-data/'
data_dir = 'datasets'
s3_bucket = 'deep-quant-data'

# we should read this list from file in datasets
remote_file = 'open-dataset-2018-03-08.dat.gz'
local_file = 'open-dataset.dat'


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
        print("File %s already exists in %s" % (filename, directory))
    return filepath


def gunzip_file(gz_path, new_path):
    """Unzips from gz_path into new_path unless it's already unzipped."""
    if not os.path.exists(new_path):
        with gzip.open(gz_path, "rb") as gz_file:
            with open(new_path, "wb") as new_file:
                for line in gz_file:
                    new_file.write(line)
        print("Unpacked %s to %s" % (gz_path, new_path))
    else:
        print("Did not unzip %s because %s already exists." % (gz_path,
                                                               new_path))


def download_data():
    print("Downloading data ...")
    maybe_download(data_dir, remote_file, data_url)
    gz_path = os.path.join(data_dir, remote_file)
    datfile_path = os.path.join(data_dir, local_file)
    gunzip_file(gz_path, datfile_path)


def main(ticlist, trimmed_datfile_name):
    open_dataset_path = os.path.join(data_dir, local_file)
    download_data()
    if ticlist is not None:
        ticlist_path = os.path.join(data_dir, ticlist)
        trimmed_datfile_path = os.path.join(data_dir, trimmed_datfile_name)
        build_and_write_trimmed_datfile(open_dataset_path,
                                        ticlist_path,
                                        trimmed_datfile_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
            "--ticlist",
            help="Specifies which tickers should be in the produced .dat "
                 "file (NOTE: please postfix this with `.dat`)")

    parser.add_argument(
            "--trimmed_datfile_name",
            help="The name you would like the produced .dat file to have. "
                 "(NOTE: please postfix this with `.dat`)")

    args = parser.parse_args()

    if args.ticlist is not None and args.trimmed_datfile_name is None:
        parser.error("Please specifiy a trimmed_datfile_name.")
    elif args.ticlist is None and args.trimmed_datfile_name is not None:
        parser.error("Please specify a ticlist.")

    main(args.ticlist, args.trimmed_datfile_name)
