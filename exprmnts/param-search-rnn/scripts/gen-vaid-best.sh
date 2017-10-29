#! /usr/bin/env bash

for NAME in $1/stdout-*.txt
do
    #    echo $NAME
    FILE=${NAME##*/}
    BASE=${FILE%.txt}
    echo $BASE
    grep MSE $NAME | cut -d ' ' -f 7,8 | sort | head -n 1 | tr '\n' ' ' > $1/valid-best-$BASE.txt
    echo $BASE | tr  '\n' ' ' >> $1/valid-best-$BASE.txt
    tail -n 2 $NAME | head -n 1 | cut -c -10 | tr '\n' ' ' >> $1/valid-best-$BASE.txt
    tail -n 1 $NAME | tr -d '\n' >> $1/valid-best-$BASE.txt 

    echo " "  >> $1/valid-best-$BASE.txt
done
