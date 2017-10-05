
CUDA_VISIBLE_DEVICES="" deep_quant.py --config=config/naive-stan-train.conf  --train=True > output/ouput-naive-stan-train.txt  2> stderr-naive.txt ;
CUDA_VISIBLE_DEVICES="" deep_quant.py --config=config/clvynt-stan-train.conf --train=True > output/ouput-clvynt-stan-train.txt 2> stderr-clvynt.txt ;
CUDA_VISIBLE_DEVICES=0  deep_quant.py --config=config/mlp-stan-train.conf    --train=True > output/ouput-mlp-stan-train.txt    2> stderr-train.txt ;

