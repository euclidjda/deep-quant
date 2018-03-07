# Experiments using WRDS data

This segment contains RNN experiment using WRDS data.

## Dataset details
Dataset is built by using the WRDS module. For details refer to the [module](https://github.com/euclidjda/deep-quant/tree/master/scripts/WRDS).

## Requirements

Make sure you are able to successfully build the dataset using WRDS module as described above. WRDS credentials are required to run the experiments.

## Running the experiments

1. Build the dataset. You can jump to Step 2 if the dataset already exists.
```shell
$ cd ~/deep-quant/exprmnts/train-model-wrds
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
$ ~/deep-quant/scripts/deep_quant.py --config=config/rnn.conf --train=True --datafile=sample_data_wrds.dat
```
The output should look like this
```shell
Loading training data ...

Setting random seed to 100
Num training entities: 1400
Num validation entities: 600
Number of batch indices: 459693
Number of batch indices: 315722
Number of batch indices: 143971

Constructing model ...
Model has the following geometry:
  model_type  = DeepRnnModel
  min_unroll  = 5
  max_unroll  = 5
  stride      = 12
  batch_size  = 128
  num_inputs  = 20
  num_outputs = 16
  num_hidden  = 128
  num_layers  = 2
  optimizer   = AdadeltaOptimizer
  device      = /gpu:0
Creating model with fresh parameters ... done in 0.60 seconds.
Calculating scaling parameters ... done in 1.32 seconds.
Training will early stop without improvement after 25 epochs.
Reading cache from ./_bcache/bcache-6857e8ea389b445edc5002df2b48741a.pkl ... done in 0.91 seconds.
Reading cache from ./_bcache/bcache-0e7691d4506f7361b9ab7fe10e0b7f33.pkl ... done in 0.41 seconds.
Steps: 1617  .................................................................................................... passes: 0.20  speed: 247 seconds
Epoch: 1 Train MSE: 44.290438 Valid MSE: 36.296103 Learning rate: 0.6000
Steps: 1617  .................................................................................................... passes: 0.20  speed: 245 seconds
Epoch: 2 Train MSE: 28.928278 Valid MSE: 25.702968 Learning rate: 0.6000
Steps: 1617  .................................................................................................... passes: 0.20  speed: 245 seconds
Epoch: 3 Train MSE: 23.222220 Valid MSE: 18.648749 Learning rate: 0.6000
Steps: 1617  .................................................................................................... passes: 0.20  speed: 245 seconds
Epoch: 4 Train MSE: 17.437683 Valid MSE: 14.027041 Learning rate: 0.6000
Steps: 1617  .................................................................................................... passes: 0.20  speed: 245 seconds
Epoch: 5 Train MSE: 14.772658 Valid MSE: 10.295938 Learning rate: 0.6000
Steps: 1617  .................................................................................................... passes: 0.20  speed: 264 seconds
Epoch: 6 Train MSE: 11.669147 Valid MSE: 7.940729 Learning rate: 0.6000
Steps: 1617  .................................................................................................... passes: 0.20  speed: 263 seconds
Epoch: 7 Train MSE: 10.737432 Valid MSE: 6.061102 Learning rate: 0.6000
Steps: 1617  .................................................................................................... passes: 0.20  speed: 267 seconds
Epoch: 8 Train MSE: 9.341028 Valid MSE: 5.028525 Learning rate: 0.6000
Steps: 1617  .................................................................................................... passes: 0.20  speed: 245 seconds
Epoch: 9 Train MSE: 8.965150 Valid MSE: 4.722991 Learning rate: 0.6000
Steps: 1617  .................................................................................................... passes: 0.20  speed: 245 seconds
Epoch: 10 Train MSE: 8.423432 Valid MSE: 4.034361 Learning rate: 0.6000
Steps: 1617  .................................................................................................... passes: 0.20  speed: 245 seconds
Epoch: 11 Train MSE: 7.190366 Valid MSE: 3.905058 Learning rate: 0.6000
Steps: 1617  .................................................................................................... passes: 0.20  speed: 244 seconds
Epoch: 12 Train MSE: 6.570809 Valid MSE: 3.865089 Learning rate: 0.6000
Steps: 1617  .................................................................................................... passes: 0.20  speed: 244 seconds
Epoch: 13 Train MSE: 7.227545 Valid MSE: 3.667518 Learning rate: 0.6000
Steps: 1617  .................................................................................................... passes: 0.20  speed: 244 seconds
Epoch: 14 Train MSE: 5.670859 Valid MSE: 3.335403 Learning rate: 0.6000
Steps: 1617  .................................................................................................... passes: 0.20  speed: 245 seconds
Epoch: 15 Train MSE: 6.203264 Valid MSE: 3.334543 Learning rate: 0.6000
Steps: 1617  .................................................................................................... passes: 0.20  speed: 245 seconds
Epoch: 16 Train MSE: 6.803460 Valid MSE: 3.635183 Learning rate: 0.6000
Steps: 1617  .................................................................................................... passes: 0.20  speed: 245 seconds
Epoch: 17 Train MSE: 5.957897 Valid MSE: 3.227542 Learning rate: 0.6000
Steps: 1617  .................................................................................................... passes: 0.20  speed: 245 seconds
Epoch: 18 Train MSE: 6.157975 Valid MSE: 3.236347 Learning rate: 0.6000
Steps: 1617  .................................................................................................... passes: 0.20  speed: 245 seconds
Epoch: 19 Train MSE: 5.889822 Valid MSE: 3.673328 Learning rate: 0.6000
Steps: 1617  .................................................................................................... passes: 0.20  speed: 245 seconds
Epoch: 20 Train MSE: 5.476445 Valid MSE: 3.133401 Learning rate: 0.6000
Steps: 1617  .................................................................................................... passes: 0.20  speed: 244 seconds
Epoch: 21 Train MSE: 5.124536 Valid MSE: 3.613623 Learning rate: 0.6000
Steps: 1617  .................................................................................................... passes: 0.20  speed: 245 seconds
Epoch: 22 Train MSE: 4.831257 Valid MSE: 3.542350 Learning rate: 0.6000
Steps: 1617  .................................................................................................... passes: 0.20  speed: 244 seconds
Epoch: 23 Train MSE: 5.524669 Valid MSE: 3.385701 Learning rate: 0.6000
Steps: 1617  .................................................................................................... passes: 0.20  speed: 244 seconds
Epoch: 24 Train MSE: 5.309324 Valid MSE: 4.217220 Learning rate: 0.6000
Steps: 1617  .................................................................................................... passes: 0.20  speed: 245 seconds
Epoch: 25 Train MSE: 6.335211 Valid MSE: 3.817821 Learning rate: 0.6000
```

3. Prediction
```shell
$ ~/deep-quant/scripts/deep_quant.py --config=config/rnn.conf --train=False --datafile=sample_data_wrds.dat --pretty_print_preds=True
```
The output should like this
```shell
Setting random seed to 521
Num training entities: 7
Num validation entities: 3
Number of batch indices: 2406

Caching batches ... done in 13.76 seconds.
Writing cache to ./_bcache/bcache-2202a1fffd4e2168306f07ae688144ad.pkl ... done in 0.24 seconds.
2018-03-04 16:39:24.694150: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
19660701 004503 mse=7140.1533
input[t-4]: 2580.00  0.00  0.00  391.00  231.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  1.00  1.00  0.00  0.00  
input[t-3]: 10742.00  0.00  0.00  1413.78  869.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.91  1.00  0.73  1.00  
input[t-2]: 11549.00  0.00  0.00  1733.18  1041.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  1.00  1.00  1.00  1.00  
input[t-1]: 12155.00  0.00  0.00  1716.00  1033.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  1.00  0.95  0.54  0.53  
input[t-0]: 13017.00  0.00  0.00  1878.00  1052.36  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.03  0.33  
output[t+1]: 4.22  1.72  0.20  0.47  0.18  0.12  0.13  0.07  0.03  0.58  0.11  0.04  0.15  0.03  0.02  0.55  
target[t+1]: 13778.60  0.00  0.00  1946.21  1080.94  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  
--------------------------------
19660901 004503 mse=7140.1777
input[t-4]: 2580.00  0.00  0.00  391.00  231.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  1.00  0.00  
input[t-3]: 10742.00  0.00  0.00  1413.78  869.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.37  1.00  0.99  
input[t-2]: 11549.00  0.00  0.00  1733.18  1041.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.06  0.72  1.00  
input[t-1]: 12155.00  0.00  0.00  1716.00  1033.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.30  
input[t-0]: 13017.00  0.00  0.00  1878.00  1052.36  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  1.00  0.84  1.00  0.39  
output[t+1]: 4.22  1.72  0.20  0.47  0.18  0.12  0.13  0.07  0.03  0.58  0.11  0.04  0.15  0.03  0.02  0.55  
target[t+1]: 13778.60  0.00  0.00  1946.21  1080.94  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
```
