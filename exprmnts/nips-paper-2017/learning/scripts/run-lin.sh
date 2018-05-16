#!/bin/bash
BIN=~/work/deep-quant/scripts

CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/rnn2-no-mom.conf --train=True --datafile=source-ml-data-100M-train.dat > output/ouput-rnn2-no-mom-train.txt    2> stderr-train.txt  ;
CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/rnn2-no-mom.conf --train=False --datafile=source-ml-data-100M.dat --pretty_print_preds=False --mse_outfile=output/mse-rnn2-no-mom-ugly.dat > output/predicts-rnn2-no-mom.dat           2> stderr-rnn2-no-mom.txt    ;
CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/rnn2-no-mom.conf --train=False --datafile=source-ml-data-100M.dat --pretty_print_preds=True  --mse_outfile=output/mse-rnn2-no-mom-pretty.dat > output/predicts-pretty-rnn2-no-mom.txt    2> stderr-rnn2-no-mom.txt    ;
