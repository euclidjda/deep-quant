BIN=~/work/euclid2/bin

for FACTOR in "ebit" "net"
do
    for MODEL in "dnn" "clvynt" "naive"
    do
	for PERIOD in "197401-199912" "200001-201608"
	do
	    echo "Running simulation for model $MODEL on $FACTOR over period $PERIOD"
	    $BIN/fundsim --config=config/sim-$FACTOR-50nms.conf \
		--log-level=4 --log-file=logging/logfile-$MODEL-$FACTOR-$PERIOD.log \
		datasets/sim-data-100M-$MODEL-$PERIOD.dat > output/output-100M-$MODEL-$FACTOR-$PERIOD.dat ;
	done	  
    done
done
