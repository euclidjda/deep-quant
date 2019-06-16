# Deep Quant

#### [by Euclidean Technologies, LLC](http://www.euclidean.com)

On a periodic basis, publicly traded companies are required to report fundamentals: financial data such as revenue, operating income, debt, among others. These data points provide some insight into the financial health of a company.

This repository contains a set of deep learning tools for forecasting company future fundamentals from historical fundamentals and other auxiliary data such as historical prices and macro economic data.

## Installation and Setup

Clone repo, setup environment, and install requirements:

```shell 
$ git clone https://github.com/euclidjda/deep-quant.git
$ cd deep-quant
$ export DEEP_QUANT_ROOT=`pwd`
$ pip install -r requirements.txt
```

You may also want to put DEEP_QUANT_ROOT in your shell initialization file such .bash_profile so that it does not need to be defined every time you start a shell. For example, you could run the following from within the deep-quant directory:

```shell 
$ echo "export DEEP_QUANT_ROOT="`pwd` >> ~/.bash_profile
```

## Preparing the Data

If you have access to 
[Wharton Research Data Services (WRDS)](https://wrds-web.wharton.upenn.edu/wrds/) 
through your academic institution, 
[go here](https://github.com/euclidjda/deep-quant/tree/master/scripts/WRDS)
to learn how to create a dataset for deep-quant.  

[WRDS instructions for creating a deep-quant data file for learning and forecasting.](https://github.com/euclidjda/deep-quant/tree/master/scripts/WRDS)

**Do not use models built with the dataset described below for actual trading 
or investing.
This is a freely available dataset assembled from freely available sources and
contains errors such as look-ahead bias and survivorship bias.**

Data is passed to `deep_quant.py` as a space-delimited
flat file. If you do not have access to a commercial or academic dataset and you would
like to test this code, we have provided a "open dataset" for this purpose.
Again this dataset should be use for testing purposes only. To obtain this file,
run the command:

```shell
$ python scripts/build_datfile.py
```

This will create a `datasets/open-dataset.dat` file.

## Building Models
You can train deep quant on a neural network of a particular type and of a
particular architecture with several other hyperparameters on a particular
dataset by first defining all of these things on a config file, and then
specifying that config file as the point of reference when running
`deep_quant.py`. Consider, for example, how deep_quant is run on
`open-dataset.dat`, as specified by `config/system-test.conf`:

```shell
$ python scripts/deep_quant.py --config=config/system-test.conf --train=True
```

This will load the corresponding data and cache it in batches in a directory
called `_bcache`, and will save model checkpoints in a directory called
`chkpts-system-test` (both of these directories will be created automatically).

A couple of notes about config files:
> * The user can specify a `.dat` file to use through the `--datafile` and the
>   `data_dir` options (note that the latter is `datasets` by default).
> * `financial_fields` is a range of columns, and should be specified as a
>   string joining the first and last columns of the `.dat` file that the user
>   wants to forecast (for example: saleq_ttm-ltq_mrq).
> * `aux_fields` is similarly also a range of columns that is equivalently
>   specified. Note, however, that these fields are strictly features; they are
>   not part of what the model is trained to predict.

## Generating Forecasts
To generate forecasts for the companies in the validation set, `deep_quant.py`
must be run with the `--train` option set to False. For example:

```shell
$ python scripts/deep_quant.py --config=config/system-test.conf --train=False --pretty_print_preds=True --mse_outfile=mse-data.txt > forecasts.txt
```

That'll produce a file called forecasts.txt with the predicted values for every
financial feature at every timestep.

## Hyper-parameter Search

The deep-quant repository contains tools for performing hyper-parameter searches. A hyper-parameter search requires a datafile
and a configuration template. A configuration template takes the format of the learning configuration file with the
exception that each configation parameter has multiple values, seperated by spaces
(for example an, see [hyper-search.conf](https://github.com/euclidjda/deep-quant/blob/master/config/hyper-search.conf)). 
The hyper-parameter search algorithms
uses the template to definine the hyper-parameter search space (i.e., all possible parameter combinations). A user may specify
one of two search algorithms: grid_search or genetic. To experiment with hyper-parameter search, execute the following 
from the deep-quant directory:

```shell
$ mkdir search
$ cp config/hyper-search.conf search/.
$ cd search
$ python $DEEP_QUANT/scripts/hyper_param_search.py --template=hyper-search.conf --search_algorithm='genetic'
```

## Uncertainty Quantification (UQ)

Forecast uncertainty is obtained by using UQ compatible deep learning models available in the _models_ directory.
Examples of the UQ parameters to be used in the config file (eg system-test.conf) are as follows:
```
--UQ                    True
--nn_type               DeepLogLikelihoodUQModel
--UQ_model_type         MVE
--df_dirname            outputs_dfs
--keep_prob_pred        0.7
```
```df_dirname``` will contain the corresponding output dataframes for prediction, data noise variance and errors.
Total variance is the sum of data noise variance (output of the NN) and model variance. Model variance is 
calculated by performing the same experiment multiple times with different random seed. Confidence Intervals can 
be calculated using the predictions and the total variance.

## Contributors and Acknowledgement

This repository was developed and is maintained by [Euclidean Technologies, LLC](http://www.euclidean.com/). Individual core contributors include [John Alberg](https://github.com/euclidjda), [Zachary Lipton](https://github.com/zackchase), [Lakshay Kumar Chauhan](https://github.com/lakshaykc), and [Ignacio Aranguren](https://github.com/nachoaz). 

## License 

This is experimental software. It is provided under the [MIT license][mit], so you can do with it whatever you wish except hold the authors responsible if it does something you don't like.

[mit]: http://www.opensource.org/licenses/mit-license.php



