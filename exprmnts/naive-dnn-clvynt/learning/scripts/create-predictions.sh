#!/bin/bash

BIN=~/work/deep-quant/scripts

CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/rnn.conf --train=False --datafile=source-ml-data-100M.dat --pretty_print_preds=True  --mse_outfile=output/mse-rnn-pretty.dat > output/predicts-pretty-rnn.txt    2> stderr-rnn.txt    ;
#CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/mlp.conf    --train=False --datafile=source-ml-data-100M.dat --pretty_print_preds=True  > output/predicts-pretty-mlp.txt    2> stderr-mlp.txt    ;
#CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/naive.conf  --train=False --datafile=source-ml-data-100M.dat --pretty_print_preds=True  > output/predicts-pretty-naive.txt  2> stderr-naive.txt  ;
#CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/clvynt.conf --train=False --datafile=source-ml-data-100M.dat --pretty_print_preds=True  > output/predicts-pretty-clvynt.txt 2> stderr-clvynt.txt ;
CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/rnn.conf --train=False --datafile=source-ml-data-100M.dat --pretty_print_preds=False --mse_outfile=output/mse-rnn-ugly.dat > output/predicts-rnn.dat           2> stderr-rnntxt    ;
# CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/mlp.conf    --train=False --datafile=source-ml-data-100M.dat --pretty_print_preds=False --mse_outfile=output/mse-mlp.dat > output/predicts-mlp.dat           2> stderr-mlp.txt    ;
#CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/naive.conf  --train=False --datafile=source-ml-data-100M.dat --pretty_print_preds=False > output/predicts-naive.dat         2> stderr-naive.txt  ;
#CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/clvynt.conf --train=False --datafile=source-ml-data-100M.dat --pretty_print_preds=False > output/predicts-clvynt.dat        2> stderr-clvynt.txt ;
