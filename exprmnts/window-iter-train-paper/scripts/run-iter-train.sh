#! /usr/bin/env bash

date

CONFIG_FILE=config/iter-default-model.conf
TRAIN_DIR=default-train
GPU=0
ORIGIN_YEAR=1980
START_YEAR=2005
END_YEAR=2017
PREDICT_TRAIN_DATA=yes
NUM_UNROLLINGS=5

while getopts c:t:s:e:p:w: option
do
    case "${option}"
	in
	c) CONFIG_FILE=${OPTARG};;
        t) TRAIN_DIR=${OPTARG};;
        s) START_YEAR=${OPTARG};;
        e) END_YEAR=${OPTARG};;
        p) PREDICT_TRAIN_DATA=${OPTARG};;
    esac
done

echo "Config: ${CONFIG_FILE}"
echo "Train dir: ${TRAIN_DIR}"
echo "GPU: /gpu:${GPU}"
echo "Start Year: ${START_YEAR}"
echo "End Year: ${END_YEAR}"
echo "Num Unrollings: ${NUM_UNROLLINGS}"
echo "Predict Train Data: ${PREDICT_TRAIN_DATA}"

ROOT=$DEEP_QUANT_ROOT
BIN=$ROOT/scripts
DATA_DIR=$ROOT/datasets
TRAIN_FILE=source-ml-data-400M.dat

CHKPTS_NAME=${TRAIN_DIR}/chkpts-train

# make training directory if it does not exist
mkdir -p ${TRAIN_DIR}

YEAR=$START_YEAR

while [ $YEAR -le $END_YEAR ]
do

    TRAIN_END=`expr ${YEAR} - 1`12
    TEST_START=${YEAR}01
    TEST_END=${YEAR}12
    TEST_START_PAD=`expr ${YEAR} - ${NUM_UNROLLINGS}`01
    TEST_END_PAD=`expr ${YEAR} + 2`12

    PROGRESS_FILE=${TRAIN_DIR}/stdout-${TEST_START}.txt

    if [ ! -e $PROGRESS_FILE ]; then
	echo -n `date +"[%m-%d %H:%M:%S]"`
	echo ": Training model with --end_date=${TRAIN_END} for test set year of ${YEAR} progress in $PROGRESS_FILE"
	$BIN/deep_quant.py --config=${CONFIG_FILE} --cache_id=1024 --datafile=${TRAIN_FILE} --train=True \
    	    --start_date=${ORIGIN_YEAR}01 --end_date=${TRAIN_END} --model_dir=${CHKPTS_NAME}-${TEST_START} > $PROGRESS_FILE
    fi

    FINAL_PREDICTIONS_FILE=${TRAIN_DIR}/test-preds-${TEST_START}.dat

    if [ ! -e $FINAL_PREDICTIONS_FILE ]; then
	echo -n `date +"[%m-%d %H:%M:%S]"`
	echo ": Creating predictions file for period ${TEST_START_PAD} to ${TEST_END_PAD}"
	$BIN/deep_quant.py --config=${CONFIG_FILE} --cache_id=9999 --datafile=${TRAIN_FILE} --train=False \
	    --start_date=${TEST_START_PAD} --end_date=${TEST_END_PAD} \
	    --model_dir=${CHKPTS_NAME}-${TEST_START} --mse_outfile=${TRAIN_DIR}/tmp-mse-${TEST_START}.dat > ${TRAIN_DIR}/tmp-pred-${TEST_START}.dat
	echo -n `date +"[%m-%d %H:%M:%S]"`
	echo ": Slicing predictions file ${TEST_START} to ${TEST_END} to create ${FINAL_PREDICTIONS_FILE}"
	$BIN/slice_data.pl $TEST_START $TEST_END < ${TRAIN_DIR}/tmp-mse-${TEST_START}.dat > ${TRAIN_DIR}/test-mse-${TEST_START}.dat
	$BIN/slice_data.pl $TEST_START $TEST_END < ${TRAIN_DIR}/tmp-pred-${TEST_START}.dat > "${FINAL_PREDICTIONS_FILE}"
    fi

    YEAR=`expr $YEAR + 1`
done

if [ $PREDICT_TRAIN_DATA == yes ]; then

    TEST_END=`expr ${START_YEAR} + 2`12 
    TEST_TAG=${START_YEAR}01
    MODEL_DIR=${CHKPTS_NAME}-${TEST_TAG}
    echo -n `date +"[%m-%d %H:%M:%S] "`
    echo ": Creating predictions for training dataset MODEL=${MODEL_DIR} TEST_END=${TEST_END}"
    $BIN/deep_quant.py --config=${CONFIG_FILE} --cache_id=9999 --datafile=${TRAIN_FILE} --train=False \
    	--start_date=${ORIGIN_YEAR}01 --end_date=${TEST_END} --model_dir=${MODEL_DIR} \
    	--mse_outfile=${TRAIN_DIR}/tmp-train-mse.dat > ${TRAIN_DIR}/tmp-train-preds.dat
    SLICE_START=${ORIGIN_YEAR}01
    SLICE_END=`expr ${START_YEAR} - 1`12
    echo -n `date +"[%m-%d %H:%M:%S] "`2 
    echo ": Slicing predictions to ${SLICE_START} - ${SLICE_END}"
    $BIN/slice_data.pl $SLICE_START $SLICE_END < ${TRAIN_DIR}/tmp-train-mse.dat > ${TRAIN_DIR}/train-mse.dat
    $BIN/slice_data.pl $SLICE_START $SLICE_END < ${TRAIN_DIR}/tmp-train-preds.dat > ${TRAIN_DIR}/train-preds.dat
fi
