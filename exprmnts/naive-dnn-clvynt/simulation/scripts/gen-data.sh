BIN=~/work/euclid2/bin

for MODEL in "mlp-stan" "clvynt-stan" "naive-stan"
do
    ./scripts/merge_model_with_simdata.pl datasets/source-data-100M.dat datasets/predicts-$MODEL.dat > datasets/merged-data-100M-$MODEL.dat; 
    $BIN/export-rescale.pl config/rescale.conf < datasets/merged-data-100M-$MODEL.dat > datasets/sim-data-100M-$MODEL.dat
done
