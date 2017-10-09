CUDA_VISIBLE_DEVICES=0 deep_quant.py --config=config/naive-stan.conf  --train=False --pretty_print_preds=True  > output/predicts-pretty-naive-stan.txt  2> stderr-naive.txt  &
CUDA_VISIBLE_DEVICES=1 deep_quant.py --config=config/clvynt-stan.conf --train=False --pretty_print_preds=True  > output/predicts-pretty-clvynt-stan.txt 2> stderr-clvynt.txt &
CUDA_VISIBLE_DEVICES=2 deep_quant.py --config=config/mlp-stan.conf    --train=False --pretty_print_preds=True  > output/predicts-pretty-mlp-stan.txt    2> stderr-mlp.txt    ;
CUDA_VISIBLE_DEVICES=0 deep_quant.py --config=config/naive-stan.conf  --train=False --pretty_print_preds=False > output/predicts-naive-stan.dat         2> stderr-naive.txt  &
CUDA_VISIBLE_DEVICES=1 deep_quant.py --config=config/clvynt-stan.conf --train=False --pretty_print_preds=False > output/predicts-clvynt-stan.dat        2> stderr-clvynt.txt &
CUDA_VISIBLE_DEVICES=2 deep_quant.py --config=config/mlp-stan.conf    --train=False --pretty_print_preds=False > output/predicts-mlp-stan.dat           2> stderr-mlp.txt
