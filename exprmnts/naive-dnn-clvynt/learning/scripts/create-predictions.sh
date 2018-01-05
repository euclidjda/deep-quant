#!/bin/bash

BIN=~/work/deep-quant/scripts

CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/rnn-v2.conf \
    --train=False --datafile=source-ml-data-v2-100M.dat --pretty_print_preds=True  \
    --mse_outfile=predicts/mse-rnn-v2-pretty.dat > predicts/predicts-rnn-v2-pretty.dat  2> stderr-rnn-v2-pretty.txt  ;

CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/rnn-v2.conf \
    --train=False --datafile=source-ml-data-v2-100M.dat --pretty_print_preds=False \
    --mse_outfile=predicts/mse-rnn2-ugly.dat > predicts/predicts-rnn-v2-ugly.dat        2> stderr-rnn-v2-ugly.txt    ;
