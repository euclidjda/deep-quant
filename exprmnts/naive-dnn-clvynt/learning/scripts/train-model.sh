#!/bin/bash

BIN=~/work/deep-quant/scripts

$BIN/deep_quant.py --config=config/dnn-stan-train.conf    --train=True > output/ouput-dnn-stan-train.txt    2> stderr-train.txt  ;
$BIN/deep_quant.py --config=config/naive-stan-train.conf  --train=True > output/ouput-naive-stan-train.txt  2> stderr-naive.txt  ;
$BIN/deep_quant.py --config=config/clvynt-stan-train.conf --train=True > output/ouput-clvynt-stan-train.txt 2> stderr-clvynt.txt ;


