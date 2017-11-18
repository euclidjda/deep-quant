#!/bin/bash
BIN=~/work/deep-quant/scripts

CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/rnn-price.conf --train=True --datafile=source-ml-data-100M-train.dat > new-predicts/ouput-rnn-price-train.txt    2> stderr-train.txt  ;
CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/rnn-price.conf --train=False --datafile=source-ml-data-100M.dat --pretty_print_preds=False --mse_outfile=new-predicts/mse-rnn-price-ugly.dat > new-predicts/predicts-rnn-price.dat           2> stderr-rnn-price.txt    ;
CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/rnn-price.conf --train=False --datafile=source-ml-data-100M.dat --pretty_print_preds=True  --mse_outfile=new-predicts/mse-rnn-price-pretty.dat > new-predicts/predicts-pretty-rnn-price.txt    2> stderr-rnn-price.txt    ;
