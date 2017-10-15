#!/bin/bash

BIN=~/work/deep-quant/scripts

$BIN/deep_quant.py --config=config/naive-stan-predict.conf  --train=False --pretty_print_preds=True  > output/predicts-pretty-naive-stan.txt  2> stderr-naive.txt  &
$BIN/deep_quant.py --config=config/clvynt-stan-predict.conf --train=False --pretty_print_preds=True  > output/predicts-pretty-clvynt-stan.txt 2> stderr-clvynt.txt &
$BIN/deep_quant.py --config=config/dnn-stan-predict.conf    --train=False --pretty_print_preds=True  > output/predicts-pretty-dnn-stan.txt    2> stderr-dnn.txt    ;
$BIN/deep_quant.py --config=config/naive-stan-predict.conf  --train=False --pretty_print_preds=False > output/predicts-naive-stan.dat         2> stderr-naive.txt  &
$BIN/deep_quant.py --config=config/clvynt-stan-predict.conf --train=False --pretty_print_preds=False > output/predicts-clvynt-stan.dat        2> stderr-clvynt.txt &
$BIN/deep_quant.py --config=config/dnn-stan-predict.conf    --train=False --pretty_print_preds=False > output/predicts-dnn-stan.dat           2> stderr-dnn.txt
