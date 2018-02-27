BIN=~/work/euclid2/bin

for MRKCAP in 400M # 100M 1B
do
    # create merge-source-$MRKCAP.dat to looks like this 
    # date gvkey ... mrkcap oiadpq_ttm
    # cut -d ' ' -f 1-12,17,22 datasets/source-data-$MRKCAP.dat > datasets/merge-source-$MRKCAP.dat
    
    for MODEL in rnn-fcst2 # naive mlp rnn lin
    do
	echo "Generating Data for $MODEL $MRKCAP"
	$BIN/rnn-merge-with-simdata.pl datasets/merge-source-$MRKCAP.dat datasets/predicts-$MODEL.dat > datasets/merged-data-$MRKCAP-$MODEL.dat; 
        # the cut cmd creates date gvkey ... mrkcap oiadpq_ttm forecast_oiadpq_ttm
	cut -d ' ' -f 1-14,18 datasets/merged-data-$MRKCAP-$MODEL.dat > datasets/stat-data-$MRKCAP-$MODEL.dat
	$BIN/slice-data.pl 200001 201708 < datasets/stat-data-$MRKCAP-$MODEL.dat > datasets/stat-data-$MRKCAP-$MODEL-200001-201708.dat
	$BIN/rnn-merge-with-consensus.pl datasets/stat-data-$MRKCAP-$MODEL-200001-201708.dat datasets/forecasts-delay-11.dat > datasets/consensus-and-$MRKCAP-$MODEL.dat
    done
done
