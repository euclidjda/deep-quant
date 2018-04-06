#!/bin/sh

cut -d ' ' -f 1,2,3,9,10 datasets/info-data-100M.dat > datasets/tmp1.dat
cut -d ' ' -f 24,39 datasets/source-data-100M.dat > datasets/tmp2.dat

paste -d ' ' datasets/tmp1.dat datasets/tmp2.dat > datasets/actuals.dat