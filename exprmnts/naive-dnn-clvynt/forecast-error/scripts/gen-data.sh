BIN=~/work/euclid2/bin

for MRKCAP in "100M" # "1B" "400M"
do
    # create merge-source-$MRKCAP.dat to looks like this 
    # date gvkey ... oiadpq_ttm
    cut -d ' ' -f 1-12,22 datasets/source-data-$MRKCAP.dat > datasets/merge-source-$MRKCAP.dat
    
    for MODEL in "rnn2" # ""lin" "naive" "mlp" "rnn" "clvynt"
    do
	echo "Generating Data for $MODEL"
	$BIN/rnn-merge-with-simdata.pl datasets/merge-source-$MRKCAP.dat datasets/predicts-$MODEL.dat > datasets/merged-data-$MRKCAP-$MODEL.dat; 
        # the cut cmd creates mom1m mom3m mom6m mom9m entval oiadpq_ttm niq_ttm, rescale then add ebit_entval and niq_entval
	cut -d ' ' -f 1-13,17-18 datasets/merged-data-$MRKCAP-$MODEL.dat > datasets/sim-data-$MRKCAP-$MODEL.dat
	$BIN/slice_data.pl 197401 199912 < datasets/sim-data-$MRKCAP-$MODEL.dat > datasets/sim-data-$MRKCAP-$MODEL-197401-199912.dat
	$BIN/slice_data.pl 200001 201612 < datasets/sim-data-$MRKCAP-$MODEL.dat > datasets/sim-data-$MRKCAP-$MODEL-200001-201612.dat
	$BIN/slice_data.pl 200001 201608 < datasets/sim-data-$MRKCAP-$MODEL.dat > datasets/sim-data-$MRKCAP-$MODEL-200001-201608.dat
    done
done
