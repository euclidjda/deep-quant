#!/bin/bash

BIN=~/work/deep-quant/scripts

CUDA_VISIBLE_DEVICES=0 $BIN/deep_quant.py --config=config/dnn.conf    --train=True --datafile=source-ml-data-100M-train.dat > output/ouput-dnn-train.txt    2> stderr-train.txt  &
CUDA_VISIBLE_DEVICES=1 $BIN/deep_quant.py --config=config/naive.conf  --train=True --datafile=source-ml-data-100M-train.dat > output/ouput-naive-train.txt  2> stderr-naive.txt  &
CUDA_VISIBLE_DEVICES=2 $BIN/deep_quant.py --config=config/clvynt.conf --train=True --datafile=source-ml-data-100M-train.dat > output/ouput-clvynt-train.txt 2> stderr-clvynt.txt 


