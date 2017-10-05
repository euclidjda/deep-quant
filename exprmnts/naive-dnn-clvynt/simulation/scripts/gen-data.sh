BIN=~/work/euclid2/bin

for MODEL in "mlp-stan" "clvynt-stan" "naive-stan"
do
    # ./scripts/merge_model_with_simdata.pl datasets/source-data-100M.dat datasets/predicts-$MODEL.dat > datasets/merged-data-100M-$MODEL.dat; 
    # cut -d ' ' -f 1-16,18,22-23 datasets/merged-data-100M-$MODEL.dat | $BIN/export-rescale.pl config/rescale.conf > datasets/sim-data-100M-$MODEL.dat
    $BIN/slice_data.pl 197801 199912 < datasets/sim-data-100M-$MODEL.dat > datasets/sim-data-100M-$MODEL-197801-199912.dat
    $BIN/slice_data.pl 200001 201609 < datasets/sim-data-100M-$MODEL.dat > datasets/sim-data-100M-$MODEL-200001-201609.dat
done
