#! /usr/bin/env bash

for i in "$@"
do
    mkdir -p train-rnn-${i}
    ./scripts/run-iter-train.sh -c config/iter-rnn-fcst.conf -t train-rnn-${i} -r $RANDOM -w 20 -s 2000 -e 2017 -p n;
done
