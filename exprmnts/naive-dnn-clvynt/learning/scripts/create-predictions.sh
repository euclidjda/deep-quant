#!/bin/bash

BIN=~/work/deep-quant/scripts

#CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/dnn.conf    --train=False --datafile=source-ml-data-100M.dat --pretty_print_preds=True  > output/predicts-pretty-dnn.txt    2> stderr-dnn.txt    ;
CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/naive.conf  --train=False --datafile=source-ml-data-100M.dat --pretty_print_preds=True  > output/predicts-pretty-naive.txt  2> stderr-naive.txt  ;
CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/clvynt.conf --train=False --datafile=source-ml-data-100M.dat --pretty_print_preds=True  > output/predicts-pretty-clvynt.txt 2> stderr-clvynt.txt ;
#CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/dnn.conf    --train=False --datafile=source-ml-data-100M.dat --pretty_print_preds=False > output/predicts-dnn.dat           2> stderr-dnn.txt    ;
CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/naive.conf  --train=False --datafile=source-ml-data-100M.dat --pretty_print_preds=False > output/predicts-naive.dat         2> stderr-naive.txt  ;
CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/clvynt.conf --train=False --datafile=source-ml-data-100M.dat --pretty_print_preds=False > output/predicts-clvynt.dat        2> stderr-clvynt.txt ;
