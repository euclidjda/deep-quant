THE TODO LIST:

- (DONE) Factor standardization use scikit learn

- (DONE) Change Y values to shifted X values for sequence learning

- (DONE) Does MSE need to be divided by batch size batch? No. Done in TF.

- (DONE) Implement Clairvioant and Naive models

- (DONE) Implement batch sequences that only require final step being an active stock.

- (DONE) Implement predicting next n-timestep average of inputs in batch_generator

- (DONE) Incorporate merge-model-with-simdata.pl into euclid2

- (DONE) Simulate clairvoyant progression from 0,3,6,12, ... months to be how perf improves

- (DONE) In predict.py, make predictions even when there is no target data available

- (DONE) Create file cache (pickle) for batch_generator

- (DONE) Add auxilary input features -- ones that are not predicted/targets (e.g., momentum)

- (DONE) Re-working scaling/unscaling implementation so it is more intuitive

- (DONE) Layer normalization in RNN

- (DONE) In predict.py, output predictions timesteps less than t. I.e., t-1, t-2, 0

- (DONE) Implement variable length sequences

- (DONE) RNN cost function upweight last k time steps instead of just last time step

- Configurable validation/holdout set methodology (holdout time window or companies)

- Implement a genetic aglorithm for hyper-parameter space search

- max-norm regularization for RNN and MLP (use tf.clip_by_norm)

- Trainable ReLu units in MLP

- rename config's nn_type to model_type

- Documentation. Starting with README.md

- Make caching faster

