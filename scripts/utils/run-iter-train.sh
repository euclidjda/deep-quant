#! /usr/bin/env bash

# Print start time
date

CONFIG_FILE=config/iter-default-model.conf
TRAIN_DIR=default-train
GPU=0
WINDOW=0
ORIGIN_YEAR=1975
START_YEAR=2005
END_YEAR=2017
PREDICT_TRAIN_DATA=yes
INIT_CHKPT="none"

while getopts c:i:t:r:o:w:s:e:p: option
do
    case "${option}"
	in
	c) CONFIG_FILE=${OPTARG};;
        i) INIT_CHKPT=${OPTARG};;
        t) TRAIN_DIR=${OPTARG};;
        r) RAND_SEED=${OPTARG};;
        o) ORIGIN_YEAR=${OPTARG};;
        w) WINDOW=${OPTARG};;
        s) START_YEAR=${OPTARG};;
        e) END_YEAR=${OPTARG};;
        p) PREDICT_TRAIN_DATA=${OPTARG};;
    esac
done

echo "Config: ${CONFIG_FILE}"
echo "Train dir: ${TRAIN_DIR}"
echo "GPU: /gpu:${GPU}"
echo "Window: ${WINDOW}"
echo "Origin Year: ${ORIGIN_YEAR}"
echo "Start Year: ${START_YEAR}"
echo "End Year: ${END_YEAR}"
echo "Predict Train Data: ${PREDICT_TRAIN_DATA}"

ROOT=$DEEP_QUANT_ROOT
BIN=$ROOT/scripts
DATA_DIR=$ROOT/datasets

CHKPTS_NAME=${TRAIN_DIR}/chkpts/chkpts-train

# make training directory if it does not exist
mkdir -p ${TRAIN_DIR}
mkdir -p ${TRAIN_DIR}/chkpts

YEAR=$START_YEAR

while [ $YEAR -le $END_YEAR ]
do

    TEST_START=${YEAR}01
    TEST_END=${YEAR}12
    TEST_START_PAD=`expr ${YEAR} - 2`01
    TEST_END_PAD=`expr ${YEAR} + 2`12

    TRAIN_END=`expr ${YEAR} - 2`12
    TRAIN_START=${ORIGIN_YEAR}01
    if [ $WINDOW -gt 0 ]; then
	TRAIN_START=`expr ${YEAR} - ${WINDOW}`01
    fi

    MODEL_DIR=${CHKPTS_NAME}-${TEST_START}

    PROGRESS_FILE=${TRAIN_DIR}/stdout-${TEST_START}.txt

    if [ -e ${INIT_CHKPT} ]; then
        echo "Pretrained model ${INIT_CHKPT} exists. Initializing with it."
    	cp -r ${INIT_CHKPT} ${MODEL_DIR}
    fi

    if [ ! -e $PROGRESS_FILE ]; then
	echo -n `date +"[%m-%d %H:%M:%S]"`
	echo ": Training model with --start_date=${TRAIN_START} --end_date=${TRAIN_END} for test set year of ${YEAR} progress in $PROGRESS_FILE"
	$BIN/deep_quant.py --config=${CONFIG_FILE} --cache_id=1024 --train=True --seed=${RAND_SEED} \
    	    --start_date=${TRAIN_START} --end_date=${TRAIN_END} --model_dir=${MODEL_DIR} > $PROGRESS_FILE
    fi

    #### INITIALIZE NEXT MODEL WITH PREVIOUS MODEL:
    #### THIS IS AN EXPERIMENTAL FEATURE THAT SHOULD BE ADDED AS A CONFIG
    # INIT_CHKPT=${MODEL_DIR}

    FINAL_PREDICTIONS_FILE=${TRAIN_DIR}/preds-${TEST_START}.dat

    echo "FINAL_PREDICTIONS_FILE=${FINAL_PREDICTIONS_FILE}"

    if [ ! -e $FINAL_PREDICTIONS_FILE ]; then
    	echo -n `date +"[%m-%d %H:%M:%S]"`
        echo ": Creating predictions file for period ${TEST_START_PAD} to ${TEST_END_PAD}"
	$BIN/deep_quant.py --config=${CONFIG_FILE} --cache_id=9999 --train=False --seed=${RAND_SEED} \
	    --start_date=${TEST_START_PAD} --end_date=${TEST_END_PAD} \
	    --model_dir=${MODEL_DIR} --mse_outfile=${TRAIN_DIR}/tmp-mse-${TEST_START}.dat > ${TRAIN_DIR}/tmp-pred-${TEST_START}.dat
	echo -n `date +"[%m-%d %H:%M:%S]"`
	echo ": Slicing predictions file ${TEST_START} to ${TEST_END} to create ${FINAL_PREDICTIONS_FILE}"
	$BIN/slice_data.pl $TEST_START $TEST_END < ${TRAIN_DIR}/tmp-mse-${TEST_START}.dat > ${TRAIN_DIR}/mse-${TEST_START}.dat
	$BIN/slice_data.pl $TEST_START $TEST_END < ${TRAIN_DIR}/tmp-pred-${TEST_START}.dat > "${FINAL_PREDICTIONS_FILE}"
    fi

    YEAR=`expr $YEAR + 1`
done

if [ $PREDICT_TRAIN_DATA == yes ]; then

    TEST_START=${ORIGIN_YEAR}01
    if [ $WINDOW -gt 0 ]; then
	TEST_START=`expr ${YEAR} - ${WINDOW}`01
    fi
    TEST_END=`expr ${START_YEAR} + 2`12 
    TEST_TAG=${START_YEAR}01
    MODEL_DIR=${CHKPTS_NAME}-${TEST_TAG}

    echo -n `date +"[%m-%d %H:%M:%S] "`
    echo ": Creating predictions for training dataset MODEL=${MODEL_DIR} TEST_END=${TEST_END}"
    $BIN/deep_quant.py --config=${CONFIG_FILE} --cache_id=9999 --train=False --seed=${RAND_SEED} \
    	--start_date=${TEST_START} --end_date=${TEST_END} --model_dir=${MODEL_DIR} \
    	--mse_outfile=${TRAIN_DIR}/tmp-train-mse.dat > ${TRAIN_DIR}/tmp-train-preds.dat
    SLICE_START=${TEST_START}
    SLICE_END=`expr ${START_YEAR} - 1`12
    echo -n `date +"[%m-%d %H:%M:%S] "`2
    echo ": Slicing predictions to ${SLICE_START} - ${SLICE_END}"
    $BIN/slice_data.pl $SLICE_START $SLICE_END < ${TRAIN_DIR}/tmp-train-mse.dat > ${TRAIN_DIR}/mse-train-${SLICE_START}-${SLICE_END}.dat
    $BIN/slice_data.pl $SLICE_START $SLICE_END < ${TRAIN_DIR}/tmp-train-preds.dat > ${TRAIN_DIR}/preds-train-${SLICE_START}-${SLICE_END}.dat
fi
