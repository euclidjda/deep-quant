
THE TODO LIST:

- Factor standardization use scikit learn (Done)

- Change Y values to shifted X values for sequence learning (DONE)

- Does MSE need to be divided by batch size batch? No. Done in TF. (DONE)

- Implement Clairvioant and Naive models (DONE)

- Simulate clairvoyant progression from 0,3,6,12, ... months to be how perf improves

- In predict.py, make predictions even when there is no target data available

- In predict.py, output make predictions timesteps less than t. I.e., t-1, t-2, 0

- Implement predicting next n-timestep average of inputs in batch_generator

- Re-working scaling/unscaling implementation so it is more intuitive

- Set passes to 0.2 for next set of trainings

- max-norm regularization for RNN and MLP (use tf.clip_by_norm)

- Create file cache for batch_generator

- Trainable ReLu units in MLP

- Batch normalization in RNN

- rename config's num_inputs to num_features

- rename config's nn_type to model_type


