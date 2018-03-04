# Experiments using WRDS data

This segment contains experiments to run experiments using WRDS data.

## Dataset details
Dataset is built by using the WRDS module. For details refer to the [module](https://github.com/euclidjda/deep-quant/tree/master/scripts/WRDS).

## Requirements

Make sure you are able to successfully build the dataset using WRDS module as described above. WRDS credentials are required to run the experiments.

## Running the experiments - RNN

1. Build the dataset
```shell
$ cd ~/deep-quant/exprmnts/train-model-wrds/rnn
$ ~/deep-quant/scripts/WRDS/build_data.py --N=10 --exclude_gics=40 --filename=~/deep-quant/datasets/sample_data_wrds.dat
```
In the above code the arguments are as follows:-
N = Number of securities sorted by market cap
exclude_gics = GICS codes to exclude from analysis
filename = The name of the output file. Make sure to save the file is saved in the datasets folder as done above.

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

Setting random seed to 521
Num training entities: 7
Num validation entities: 3
Number of batch indices: 2310
Number of batch indices: 1323
Number of batch indices: 987
2018-03-04 16:13:13.812119: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA

Constructing model ...
Model has the following geometry:
  model_type  = DeepRnnModel
  min_unroll  = 5
  max_unroll  = 5
  batch_size  = 256
  num_inputs  = 20
  num_outputs = 16
  num_hidden  = 64
  num_layers  = 2
  optimizer   = AdadeltaOptimizer
  device      = /gpu:0
Creating model with fresh parameters ... done in 0.55 seconds.
Calculating scaling parameters ... done in 0.75 seconds.
Training will early stop without improvement after 25 epochs.
Reading cache from ./_bcache/bcache-f325edd65346188b40e0defa66841017.pkl ... done in 0.00 seconds.
Reading cache from ./_bcache/bcache-6388dd54f750ab6f5c840232b2495031.pkl ... done in 0.00 seconds.
Steps: 4  .................................................................................................... passes: 0.20  speed: 2 seconds
Epoch: 1 Train MSE: 282.490936 Valid MSE: 195.845795 Learning rate: 0.6000
Creating directory chkpts/chkpts-rnn
Steps: 4  .................................................................................................... passes: 0.20  speed: 0 seconds
Epoch: 2 Train MSE: 282.117676 Valid MSE: 195.487956 Learning rate: 0.6000
Steps: 4  .................................................................................................... passes: 0.20  speed: 0 seconds
Epoch: 3 Train MSE: 281.473175 Valid MSE: 195.199361 Learning rate: 0.6000
Steps: 4  .................................................................................................... passes: 0.20  speed: 0 seconds
Epoch: 4 Train MSE: 281.821686 Valid MSE: 194.938619 Learning rate: 0.6000
Steps: 4  .................................................................................................... passes: 0.20  speed: 0 seconds
Epoch: 5 Train MSE: 280.885315 Valid MSE: 194.534856 Learning rate: 0.6000
Steps: 4  .................................................................................................... passes: 0.20  speed: 0 seconds
Epoch: 6 Train MSE: 281.497986 Valid MSE: 194.295314 Learning rate: 0.6000
Steps: 4  .................................................................................................... passes: 0.20  speed: 0 seconds
Epoch: 7 Train MSE: 281.097595 Valid MSE: 194.044937 Learning rate: 0.6000
Steps: 4  .................................................................................................... passes: 0.20  speed: 0 seconds
Epoch: 8 Train MSE: 281.298340 Valid MSE: 193.841476 Learning rate: 0.6000
Steps: 4  .................................................................................................... passes: 0.20  speed: 0 seconds
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
