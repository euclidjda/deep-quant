#!/bin/bash

BIN=~/work/deep-quant/scripts

#CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/rnn.conf --train=True --datafile=source-ml-data-100M-train.dat > output/ouput-rnn-train.txt    2> stderr-train.txt  ;
#CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/rnn2.conf --train=True --datafile=source-ml-data-100M-train.dat > output/ouput-rnn2-train.txt    2> stderr-train.txt  ;
#CUDA_VISIBLE_DEVICES=""   $BIN/deep_quant.py --config=config/mlp.conf    --train=True --datafile=source-ml-data-100M-train.dat > output/ouput-mlp-train.txt    2> stderr-train.txt  ;
CUDA_VISIBLE_DEVICES=""   $BIN/deep_quant.py --config=config/naive.conf  --train=True --datafile=source-ml-data-100M-train.dat > output/ouput-naive-train.txt  2> stderr-naive.txt  ;
#CUDA_VISIBLE_DEVICES=""   $BIN/deep_quant.py --config=config/clvynt.conf --train=True --datafile=source-ml-data-100M-train.dat > output/ouput-clvynt-train.txt 2> stderr-clvynt.txt 

