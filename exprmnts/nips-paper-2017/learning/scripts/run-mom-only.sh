#!/bin/bash

BIN=~/work/deep-quant/scripts

CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/rnn2-mom-only.conf --train=True --datafile=source-ml-data-100M-train.dat \
    > output/ouput-rnn2-mom-only-train.txt \
    2> stderr-train.txt  ;

CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/rnn2-mom-only.conf --train=False --datafile=source-ml-data-100M.dat \
    --pretty_print_preds=False --mse_outfile=output/mse-rnn2-mom-only-ugly.dat \
    > output/predicts-rnn2-mom-only.dat \
    2> stderr-rnn2-mom-only.txt    ;

CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/rnn2-mom-only.conf --train=False --datafile=source-ml-data-100M.dat \
    --pretty_print_preds=True  --mse_outfile=output/mse-rnn2-mom-only-pretty.dat \
    > output/predicts-pretty-rnn2-mom-only.txt \
    2> stderr-rnn2-mom-only.txt    ;
