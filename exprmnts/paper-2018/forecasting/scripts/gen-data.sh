#!/bin/sh

for MRKCAP in 100M
do
    cut -d ' ' -f 1,2,3,9,10 datasets/info-data-${MRKCAP}.dat > datasets/tmp1.dat
    cut -d ' ' -f 24,39 datasets/source-data-${MRKCAP}.dat > datasets/tmp2.dat
    paste -d ' ' datasets/tmp1.dat datasets/tmp2.dat > datasets/actuals-${MRKCAP}.dat
done

for MODEL in rnn-iter-w20 rnn-iter-w20-sqshr
do
    for MRKCAP in 100M
    do
	scripts/raw-preds-to-preds.py datasets/actuals-${MRKCAP}.dat datasets/merged-raw-preds-${MODEL}.dat \
	    > datasets/forecasts-${MODEL}-${MRKCAP}.dat
    done
done
