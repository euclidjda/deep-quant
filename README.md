# Deep Quant

#### [by Euclidean Technologies, LLC](http://www.euclidean.com)

Deep learning on company fundamental data for long-term investing

## Installation and Setup

Clone repo, setup environment, and install requirements:

```shell 
$ git clone https://github.com/euclidjda/deep-quant.git
$ cd deep-quant
$ export DEEP_QUANT_ROOT=`pwd`
$ pip install -r requirements.txt
```

## Preparing the Data

If you have access to 
[Wharton Research Data Services (WRDS)](https://wrds-web.wharton.upenn.edu/wrds/) 
through your academic institution, 
[go here](https://github.com/euclidjda/deep-quant/tree/master/scripts/WRDS)
to learn how to create a dataset for deep-quant.  

[WRDS instructions for creating a deep-quant data file for learning and forecasting.](https://github.com/euclidjda/deep-quant/tree/master/scripts/WRDS)

**Do not use models built with the dataset described below for actually trading 
or investing.
This is a freely available dataset assembled from freely available sources and
contains errors such as look-ahead bias and survivorship bias.**

Data is passed to `deep_quant.py` as a `.dat` file, which is a space-delimited
file. The user can either run deep-quant on the full `open-dataset.dat` that's
provided, or --if, for example, the user wants to train a model on a particular
set of tickers-- on a trimmed version of `open-dataset.dat`. To obtain this
file, run the command:

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
$ python scripts/deep_quant.py --config/system-test.conf --train=True
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

## Contributors and Acknowledgement

This repository was developed and is maintained by [Euclidean Technologies, LLC](http://www.euclidean.com/). 

## License 

This is experimental software. It is provided under the [MIT license][mit], so you can do with it whatever you wish except hold the authors responsible if it does something you don't like.

[mit]: http://www.opensource.org/licenses/mit-license.php



