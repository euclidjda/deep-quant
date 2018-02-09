# WRDS DATA

This module builds the dataset from WRDS used by [deep-quant](https://github.com/euclidjda/deep-quant) . WRDS account ceredentials are required to run the module.

## Dataset details
Fundamental data for top N market cap equities is acquired from WRDS and processed to be used for training. Configuration for data acquition is described in usage.

## Features
active,		date,		gvkey,		year,  		month,		
mom1m,		mom3m,		mom6m,		mom9m,		mrkcap,
entval,		saleq_ttm,	cogsq_ttm,	xsgaq_ttm,	oiadpq_ttm,
niq_ttm,	cheq_mrq,	rectq_mrq,	invtq_mrq,	acoq_mrq,
ppentq_mrq,	aoq_mrq,	dlcq_mrq,	apq_mrq,	txpq_mrq,
lcoq_mrq,   ltq_mrq,	csho_1yr_avg

## Python Libary Requirements
1. Numpy
2. Pandas
3. Psycopg2
4. WRDS - 	Github repo can be found [here](https://github.com/wharton/wrds). 
			WRDS requires Pandas and Psycopg2. Details of Psycopg2 are given on the linked page.
			
## Additional Requirements
1. WRDS account credentials

## Usage
1. build_data_config.ini

	N = Number of securities sorted by market cap
	
	Exclude GICS Codes - GICS industry codes to be removed for analysis

	*No need to edit gvkey_config.ini*
	
2. gvkey_config.ini

	No need to edit this file. It is used to keep track of equities from previous runs for contiuation.
	
	Caveat: most_recent_list in the file will be populated after every run. If you want to run a test with smaller N after running a larger N, make sure to clear the list. You may delete everything from the file except the first line which defines the section for the config file.

3. Run build_data.py

	Provide username and password for WRDS account. The prompt will ask for username and password everytime build_data.py is run. 
	
	Not required but if you want to automate it, username and password can be saved in the postgresql config file. Details are [here](https://www.postgresql.org/docs/9.3/static/libpq-pgpass.html)
	
4. Dataset is saved as a csv