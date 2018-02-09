# WRDS DATA

This module builds the dataset from WRDS used by [deep-quant](https://github.com/euclidjda/deep-quant) . WRDS account ceredentials are required to run the module.

## Dataset details
Fundamental data for top N market cap equities is acquired from WRDS and processed to be used for training. Configuration for data acquition is described in usage.

### Features in the dataset
active,		date,		gvkey,		year,  		month,		
mom1m,		mom3m,		mom6m,		mom9m,		mrkcap,
entval,		saleq_ttm,	cogsq_ttm,	xsgaq_ttm,	oiadpq_ttm,
niq_ttm,	cheq_mrq,	rectq_mrq,	invtq_mrq,	acoq_mrq,
ppentq_mrq,	aoq_mrq,	dlcq_mrq,	apq_mrq,	txpq_mrq,
lcoq_mrq,   ltq_mrq,	csho_1yr_avg

## Setup and Installation
Make sure the following packages are installed.

1. Numpy
2. Pandas
3. Psycopg2

You may install the above three prerequisites as follows:
```shell
$ cd deep-quant/scripts/WRDS
$ sudo pip install -r requirements.txt
```

4. WRDS - 	This cannot be installed using pip. Install from the Github repository found [here](https://github.com/wharton/wrds). Make sure above prerequisites are satisfied before installing WRDS.

Clone the repository with:
```shell
$ git clone https://github.com/wharton/wrds.git
$ cd wrds-master
$ python setup.py install
```						
Test WRDS

```shell
>>> import wrds
>>> db = wrds.Connection()
Enter your credentials.
Username: <your_username>
Password: <your_password>
>>> db.list_libraries()
['audit', 'bank', 'block', 'bvd', 'bvdtrial', 'cboe', ...]
>>> db.list_tables(library='crsp')
['aco_amda', 'aco_imda', 'aco_indfnta', 'aco_indfntq', ...]
>>> db.describe_table(library='csrp', table='stocknames')
Approximately 58957 rows in crsp.stocknames.
       name    nullable              type
0      permno      True  DOUBLE PRECISION      
1      permco      True  DOUBLE PRECISION      
2      namedt      True              DATE
...

>>> stocknames = db.get_table(library='crsp', table='stocknames', obs=10)
>>> stocknames.head()
   permno  permco      namedt   nameenddt     cusip    ncusip ticker  \
0  10000.0  7952.0  1986-01-07  1987-06-11  68391610  68391610  OMFGA
1  10001.0  7953.0  1986-01-09  1993-11-21  36720410  39040610   GFGC
2  10001.0  7953.0  1993-11-22  2008-02-04  36720410  29274A10   EWST
3  10001.0  7953.0  2008-02-05  2009-08-03  36720410  29274A20   EWST
4  10001.0  7953.0  2009-08-04  2009-12-17  36720410  29269V10   EGAS
...
```

## Building data
Build data as follows:
```shell
$ cd deep-quant/scripts/WRDS
$ python build_data.py --N 10 --exclude_gics 40,45 --filename out.dat --test_mode yes
```

The above command will prompt you for username and password. Terminal output should look like
```shell
Enter your WRDS username [lakshay]:<your_username>
Enter your password:<your_passwrod>
WRDS recommends setting up a .pgpass file.
You can find more info here:
https://www.postgresql.org/docs/9.5/static/libpq-pgpass.html.
Loading library list...
Done

Shape of raw dataframe: 533,21

Total Number of Equities in the dataset : 3

Total Execution Time: 21.29
```

To avoid entering username and password everytime, you can setup your username and password in .pgpass file. Details can be found [here](https://www.postgresql.org/docs/9.3/static/libpq-pgpass.html).

build_data.py uses arguments as below:

- N : Number of equities sorted by market cap. Top N equities are considered.
- exclude_gics : Excludes the industries with list of GICS codes
- filename : Name of the output file
- test_mode : Run in test mode when N is smaller. Default value is no

*Note: Companies from previous runs are retained for continuation. It is recommended to run in test_mode if not building the actual dataset*

The output data file shall look like as follows:
```shell
active date gvkey year month mom1m mom3m mom6m mom9m mrkcap ...
1.0 19630701 006266 1963.0 7.0 0.0 0.0 1.0 0.0 0.0 .0 93.7 ...
1.0 19630801 006266 1963.0 8.0 0.0 1.0 0.0 0.0 0.0 .0 93.7 ...
1.0 19630901 006266 1963.0 9.0 1.0 1.0 0.0 1.0 0.0 .0 93.7 ...
1.0 19631001 006266 1963.0 10.0 1.0 1.0 1.0 1.0 0.00.0 184.6 ...
1.0 19631101 006266 1963.0 11.0 1.0 1.0 1.0 1.0 0.00.0 184.6 ...
1.0 19631201 006266 1963.0 12.0 0.0 1.0 1.0 1.0 0.00.0 184.6 ...
1.0 19640101 006266 1964.0 1.0 0.0 0.0 1.0 1.0 0.0 .0 277.8 ...
1.0 19640201 006266 1964.0 2.0 0.0 0.0 1.0 1.0 0.0 .0 277.8 ...
1.0 19640301 006266 1964.0 3.0 1.0 0.0 1.0 1.0 0.0 .0 277.8 ...
1.0 19640401 006266 1964.0 4.0 0.0 0.0 0.0 0.0 0.0 .0 364.8 ...
1.0 19640501 006266 1964.0 5.0 0.0 0.0 0.0 0.0 0.0 .0 364.8 ...
1.0 19640601 006266 1964.0 6.0 1.0 0.0 0.0 0.0 0.0 .0 364.8 ...
...
...
...
```
