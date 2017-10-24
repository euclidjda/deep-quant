#! /usr/bin/env bash

START_YEAR=2000
END_YEAR=2015
YEAR=$START_YEAR

echo $1

while [ $YEAR -le $END_YEAR ]
do

    grep error $1/stdout-${YEAR}01.txt | cut -d ' ' -f 5,7,8 | sort | head -n 1 > $1/valid-best-${YEAR}.txt
    
    YEAR=`expr $YEAR + 1`

done
