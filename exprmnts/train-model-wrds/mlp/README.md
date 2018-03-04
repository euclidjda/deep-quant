# Experiments using WRDS data

This segment contains MLP experiment using WRDS data.

## Dataset details
Dataset is built by using the WRDS module. For details refer to the [module](https://github.com/euclidjda/deep-quant/tree/master/scripts/WRDS).

## Requirements

Make sure you are able to successfully build the dataset using WRDS module as described above. WRDS credentials are required to run the experiments.

## Running the experiments

1. Build the dataset. You can jump to Step 2 if the dataset already exists.
```shell
$ cd ~/deep-quant/exprmnts/train-model-wrds/mlp
$ ~/deep-quant/scripts/WRDS/build_data.py --N=10 --exclude_gics=40 --filename=~/deep-quant/datasets/sample_data_wrds.dat
```
In the above code the arguments are as follows:-

- N = Number of securities sorted by market cap
- exclude_gics = GICS codes to exclude from analysis
- filename = The name of the output file. Make sure to save the file is saved in the datasets folder as done above.

You will be prompted to enter your WRDS credentials.

The output should look like this
```shell
Enter your WRDS username [username]:<your_username>
Enter your password: <your_password>
WRDS recommends setting up a .pgpass file.
You can find more info here:
https://www.postgresql.org/docs/9.5/static/libpq-pgpass.html.
Loading library list...
Done


Shape of raw dataframe: 1304,21

---------------------------------------------------------------------------------
Total Number of Equities in the dataset: 10

Total Execution Time: 30.65
```

2. Train on the built dataset
```shell
$ mkdir chkpts
$ ~/deep-quant/scripts/deep_quant.py --config=config/mlp.conf --train=True --datafile=sample_data_wrds.dat
```

The output should like this
```shell
Loading training data ...

Setting random seed to 521
Num training entities: 7
Num validation entities: 3
Number of batch indices: 2310
Number of batch indices: 1323
Number of batch indices: 987
2018-03-04 17:11:24.244653: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA

Constructing model ...
Model has the following geometry:
  model_type  = DeepMlpModel
  min_unroll  = 5
  max_unroll  = 5
  batch_size  = 256
  num_inputs  = 20
  num_outputs = 16
  num_hidden  = 1024
  num_layers  = 2
  optimizer   = AdadeltaOptimizer
  device      = /gpu:0
Creating model with fresh parameters ... done in 0.08 seconds.
Calculating scaling parameters ... done in 0.09 seconds.
Training will early stop without improvement after 25 epochs.

Caching batches ... done in 6.78 seconds.
Writing cache to ./_bcache/bcache-f325edd65346188b40e0defa66841017.pkl ... done in 0.03 seconds.

Caching batches ... done in 4.01 seconds.
Writing cache to ./_bcache/bcache-6388dd54f750ab6f5c840232b2495031.pkl ... done in 0.02 seconds.
Steps: 4  .................................................................................................... passes: 0.20  speed: 0 seconds
Epoch: 1 Train MSE: 282.752533 Valid MSE: 194.521556 Learning rate: 0.6000
Creating directory chkpts/chkpts-mlp
Steps: 4  .................................................................................................... passes: 0.20  speed: 0 seconds
Epoch: 2 Train MSE: 277.830383 Valid MSE: 192.587809 Learning rate: 0.6000
Steps: 4  .................................................................................................... passes: 0.20  speed: 0 seconds
Epoch: 3 Train MSE: 275.607605 Valid MSE: 190.656672 Learning rate: 0.6000
Steps: 4  .................................................................................................... passes: 0.20  speed: 0 seconds
Epoch: 4 Train MSE: 272.543610 Valid MSE: 188.827850 Learning rate: 0.6000
Steps: 4  .................................................................................................... passes: 0.20  speed: 0 seconds
Epoch: 5 Train MSE: 267.018921 Valid MSE: 187.002823 Learning rate: 0.6000
Steps: 4  .................................................................................................... passes: 0.20  speed: 0 seconds
Epoch: 6 Train MSE: 265.008484 Valid MSE: 185.202204 Learning rate: 0.6000
Steps: 4  .................................................................................................... passes: 0.20  speed: 0 seconds
Epoch: 7 Train MSE: 261.172363 Valid MSE: 183.317164 Learning rate: 0.6000
Steps: 4  .................................................................................................... passes: 0.20  speed: 0 seconds
Epoch: 8 Train MSE: 257.681854 Valid MSE: 181.389430 Learning rate: 0.6000
```

3. Prediction
```shell
$ ~/deep-quant/scripts/deep_quant.py --config=config/mlp.conf --train=False --datafile=sample_data_wrds.dat --pretty_print_preds=True
```
The output should like this
```shell

Setting random seed to 521
Num training entities: 7
Num validation entities: 3
Number of batch indices: 2406

Caching batches ... done in 13.65 seconds.
Writing cache to ./_bcache/bcache-2202a1fffd4e2168306f07ae688144ad.pkl ... done in 0.25 seconds.
2018-03-04 17:19:34.142647: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
19660701 004503 mse=7200.7588
input[t-4]: 2580.00  0.00  0.00  391.00  231.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  1.00  1.00  0.00  0.00  
input[t-3]: 10742.00  0.00  0.00  1413.78  869.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.91  1.00  0.73  1.00  
input[t-2]: 11549.00  0.00  0.00  1733.18  1041.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  1.00  1.00  1.00  1.00  
input[t-1]: 12155.00  0.00  0.00  1716.00  1033.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  1.00  0.95  0.54  0.53  
input[t-0]: 13017.00  0.00  0.00  1878.00  1052.36  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.03  0.33  
output[t+1]: 0.60  0.08  0.18  0.43  0.10  0.08  -0.00  0.02  0.01  0.19  0.06  0.03  0.02  0.01  0.01  0.30  
target[t+1]: 13778.60  0.00  0.00  1946.21  1080.94  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  
--------------------------------
19660801 004503 mse=7209.9082
input[t-4]: 2580.00  0.00  0.00  391.00  231.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  1.00  1.00  0.00  
input[t-3]: 10742.00  0.00  0.00  1413.78  869.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  1.00  0.83  1.00  1.00  
input[t-2]: 11549.00  0.00  0.00  1733.18  1041.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.59  1.00  0.89  1.00  
input[t-1]: 12155.00  0.00  0.00  1716.00  1033.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.46  0.04  0.40  
input[t-0]: 13017.00  0.00  0.00  1878.00  1052.36  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  1.00  0.00  0.23  0.18  
output[t+1]: 0.62  0.13  0.18  0.42  0.10  0.09  0.01  0.02  0.01  0.16  0.05  0.02  0.03  0.01  0.01  0.24  
target[t+1]: 13778.60  0.00  0.00  1946.21  1080.94  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  
--------------------------------
```
