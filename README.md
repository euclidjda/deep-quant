# deepquant
Deep learning on company fundamental data for long-term investing

# Installation and Setup
This might be facilitated using a Docker container in the future.
For now:
Create a virtual environment
`conda create --name env`
or
`virtualenv env`
Activate that virtual environment
`source activate env`
Install prerequisites:
`pip install -r requirements.txt`

Please also add DEEP\_QUANT\_ROOT to your environment:
`export DEEP_QUANT_ROOT=/path/to/deep-quant/in/your/machine`

# Running the System Test
`python scripts/deep_quant.py --config=config/system_test.conf --Train=True`
