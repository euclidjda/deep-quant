BIN=~/work/euclid2/bin

for MODEL in "mlp-stan" "clvynt-stan" "naive-stan"
do
    cut -d ' ' -f 1-18 datasets/source-data-100M.dat > datasets/merge-source-100M.dat
    $BIN/rnn-merge-with-simdata.pl datasets/merge-source-100M.dat datasets/predicts-$MODEL.dat > datasets/merged-data-100M-$MODEL.dat; 
    # the cut cmd creates mom1m mom3m mom6m mom9m entval oiadpq_ttm niq_ttm, rescale then add ebit_entval and niq_entval
    cut -d ' ' -f 1-16,18,22-23 datasets/merged-data-100M-$MODEL.dat | $BIN/export-rescale.pl config/rescale.conf > datasets/sim-data-100M-$MODEL.dat
    $BIN/slice_data.pl 197401 199912 < datasets/sim-data-100M-$MODEL.dat > datasets/sim-data-100M-$MODEL-197401-199912.dat
    $BIN/slice_data.pl 200001 201608 < datasets/sim-data-100M-$MODEL.dat > datasets/sim-data-100M-$MODEL-200001-201608.dat
done
