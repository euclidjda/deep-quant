# deepquant
Deep learning on company fundamental data for long-term investing

## Installation and Setup

Clone repo, setup environment, and install requirements:

```shell 
git clone https://github.com/euclidjda/deep-quant.git
cd deep-quant
export DEEP_QUANT_ROOT=`pwd`
pip3 install -r requirements.txt
```

## Running the System Test

`python scripts/deep_quant.py --config=config/system_test.conf --train=True`

or, for python3:

`python3 scripts/deep_quant.py --config=config/system_test.conf --train=True`
