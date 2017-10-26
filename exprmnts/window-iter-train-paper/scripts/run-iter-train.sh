#! /usr/bin/env bash

date

CONFIG_FILE=config/iter-rnn.conf
TRAIN_DIR=train-rnn
GPU=0
START_YEAR=2000
END_YEAR=2017
PREDICT_TRAIN_DATA=yes
WINDOW_SIZE=25
NUM_UNROLLINGS=5

while getopts c:t:s:e:p: option
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
echo "Predict Train Data: ${PREDICT_TRAIN_DATA}"

ROOT=$DEEP_QUANT_ROOT
BIN=$ROOT/scripts
DATA_DIR=$ROOT/datasets
TRAIN_FILE=source-ml-data-100M.dat

CHKPTS_NAME=${TRAIN_DIR}/chkpts-train

# make training directory if it does not exist
mkdir -p ${TRAIN_DIR}

YEAR=$START_YEAR

while [ $YEAR -le $END_YEAR ]
do
    TEST_START_PRE=`expr ${YEAR} - ${NUM_UNROLLINGS}`01
    TEST_START=${YEAR}01
    TEST_END=${YEAR}12
    TRAIN_START=`expr ${YEAR} - ${WINDOW_SIZE} - ${NUM_UNROLLINGS}`01
    TRAIN_END=`expr ${YEAR} - 1`12

    PROGRESS_FILE=${TRAIN_DIR}/stdout-${TEST_START}.txt

    if [ ! -e $PROGRESS_FILE ]; then
	echo -n `date +"[%m-%d %H:%M:%S]"`
	echo ": Training model on ${TRAIN_START} to ${TRAIN_END} for test set year of ${YEAR} progress in $PROGRESS_FILE"
	$BIN/deep_quant.py --config=${CONFIG_FILE} --datafile=${TRAIN_FILE} --train=True \
    	    --start_date=${TRAIN_START} --end_date=${TRAIN_END} --model_dir=${CHKPTS_NAME}-${TEST_START} > $PROGRESS_FILE
	#cp -rf ${CHKPTS_NAME} ${CHKPTS_NAME}-${TEST_START}
    fi

    FINAL_PREDICTIONS_FILE=${TRAIN_DIR}/test-preds-${TEST_START}.dat

    if [ ! -e $FINAL_PREDICTIONS_FILE ]; then
	echo -n `date +"[%m-%d %H:%M:%S]"`
	echo ": Creating predictions file for period ${TEST_START_PRE} to ${TEST_END}"
	$BIN/deep_quant.py --config=${CONFIG_FILE} --datafile=${TRAIN_FILE} --train=False \
            --start_date=${TEST_START_PRE} --end_date=${TEST_END} --model_dir=${CHKPTS_NAME}-${TEST_START} > ${TRAIN_DIR}/tmp-${TEST_START}.dat
	echo -n `date +"[%m-%d %H:%M:%S]"`
	echo ": Slicing predictions file ${TEST_START} to ${TEST_END} to create ${FINAL_PREDICTIONS_FILE}"
	$BIN/slice_data.pl $TEST_START $TEST_END < ${TRAIN_DIR}/tmp-${TEST_START}.dat > "${FINAL_PREDICTIONS_FILE}"
    fi

    YEAR=`expr $YEAR + 1`
done

if [ $PREDICT_TRAIN_DATA == yes ]; then

    echo -n `date +"[%m-%d %H:%M:%S] "`
    echo ": Creating predictions for training dataset"

    TEST_END=`expr ${START_YEAR} - 1`12 
    TEST_TAG=${START_YEAR}01

    $BIN/deep_quant.py --config=${CONFIG_FILE} --datafile=${TRAIN_FILE} --train=False \
     --end_date=${TEST_END} --model_dir=${CHKPTS_NAME}-${TEST_TAG} > ${TRAIN_DIR}/train-preds.dat
fi
