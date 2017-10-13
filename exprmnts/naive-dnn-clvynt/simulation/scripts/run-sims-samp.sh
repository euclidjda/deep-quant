BIN=~/work/euclid2/bin

for MODEL in "mlp-stan" "clvynt-stan" "naive-stan"
do
    for PERIOD in "197401-199912" "200001-201609"
    do
	echo "Running simulation for model $MODEL over period $PERIOD"
	$BIN/fundsim --config=config/sim-samp-50nms.conf datasets/sim-data-100M-$MODEL-$PERIOD.dat \
	    --dump-file=logging/dump-$MODEL-$PERIOD.dat --log-file=logging/logfile-$MODEL-$PERIOD.log > output/output-100M-$MODEL-$PERIOD.dat ;
    done	  
done
