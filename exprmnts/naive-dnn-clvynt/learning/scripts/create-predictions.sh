#!/bin/bash

BIN=~/work/deep-quant/scripts

#CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/rnn2.conf --train=False --datafile=source-ml-data-100M.dat --pretty_print_preds=True  --mse_outfile=new-predicts/mse-rnn2-pretty.dat > new-predicts/predicts-pretty-rnn2.txt    2> stderr-rnn2.txt    ;
#CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/mlp.conf    --train=False --datafile=source-ml-data-100M.dat --pretty_print_preds=True  > new-predicts/predicts-pretty-mlp.txt    2> stderr-mlp.txt    ;
#CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/naive.conf  --train=False --datafile=source-ml-data-100M.dat --pretty_print_preds=True  > new-predicts/predicts-pretty-naive.txt  2> stderr-naive.txt  ;

CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/rnn2.conf --train=False --datafile=source-ml-data-100M.dat --pretty_print_preds=False --mse_outfile=new-predicts/mse-rnn2-ugly.dat > new-predicts/predicts-rnn2.dat           2> stderr-rnn2txt    ;
CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/rnn.conf --train=False --datafile=source-ml-data-100M.dat --pretty_print_preds=False --mse_outfile=new-predicts/mse-rnn-ugly.dat > new-predicts/predicts-rnn.dat           2> stderr-rnntxt    ;
 CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/mlp.conf    --train=False --datafile=source-ml-data-100M.dat --pretty_print_preds=False --mse_outfile=new-predicts/mse-mlp.dat > new-predicts/predicts-mlp.dat           2> stderr-mlp.txt    ;
CUDA_VISIBLE_DEVICES="" $BIN/deep_quant.py --config=config/naive.conf  --train=False --datafile=source-ml-data-100M.dat --pretty_print_preds=False --mse_outfile=new-predictss/mse-naive.dat > new-predicts/predicts-naive.dat         2> stderr-naive.txt  ;
