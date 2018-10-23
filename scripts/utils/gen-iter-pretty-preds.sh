#! /usr/bin/env bash

date

CONFIG_FILE=config/iter-default-model.conf
TRAIN_DIR=default-train
GPU=0
START_YEAR=2005
END_YEAR=2017

while getopts c:t:r:s:e: option
do
    case "${option}"
	in
	c) CONFIG_FILE=${OPTARG};;
        t) TRAIN_DIR=${OPTARG};;
        r) RAND_SEED=${OPTARG};;
        s) START_YEAR=${OPTARG};;
        e) END_YEAR=${OPTARG};;
    esac
done

echo "Config: ${CONFIG_FILE}"
echo "Train dir: ${TRAIN_DIR}"
echo "GPU: /gpu:${GPU}"
echo "Start Year: ${START_YEAR}"
echo "End Year: ${END_YEAR}"

ROOT=$DEEP_QUANT_ROOT
BIN=$ROOT/scripts
DATA_DIR=$ROOT/datasets

CHKPTS_NAME=${TRAIN_DIR}/chkpts/chkpts-train

YEAR=$START_YEAR

while [ $YEAR -le $END_YEAR ]
do

    TEST_START=${YEAR}01
    TEST_END=${YEAR}12
    TEST_END_PAD=`expr ${YEAR} + 1`12

    MODEL_DIR=${CHKPTS_NAME}-${TEST_START}

    FINAL_PREDICTIONS_FILE=${TRAIN_DIR}/pretty-preds-${TEST_START}.dat

    echo "FINAL_PREDICTIONS_FILE=${FINAL_PREDICTIONS_FILE}"

    if [ ! -e $FINAL_PREDICTIONS_FILE ]; then
    	echo -n `date +"[%m-%d %H:%M:%S]"`
        echo ": Creating predictions file for period ${TEST_START} to ${TEST_END}"
	$BIN/deep_quant.py --config=${CONFIG_FILE} --cache_id=9999 --train=False \
	    --start_date=${TEST_START} --end_date=${TEST_END_PAD} \
	    --pretty_print_preds=True --require_targets=True \
	    --mse_outfile=${TRAIN_DIR}/pretty-mse-${TEST_START}.dat \
	    --model_dir=${MODEL_DIR}  >  "${FINAL_PREDICTIONS_FILE}"
    fi

    YEAR=`expr $YEAR + 1`
done
