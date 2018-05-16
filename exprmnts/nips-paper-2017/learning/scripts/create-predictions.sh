#!/bin/bash

BIN=~/work/deep-quant/scripts

CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/naive.conf \
    --train=False --datafile=source-ml-data-100M.dat --pretty_print_preds=True  \
    --mse_outfile=output/mse-naive-pretty.dat > output/predicts-naive-pretty.dat  2> stderr-naive-pretty.txt  ;

CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/naive.conf \
    --train=False --datafile=source-ml-data-100M.dat --pretty_print_preds=False \
    --mse_outfile=output/mse-naive-ugly.dat > output/predicts-naive-ugly.dat        2> stderr-naive-ugly.txt    ;
