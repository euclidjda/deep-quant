"""
Builds the fundamental dataset for top N market cap equitities from WRDS.
Requires WRDS account. Enter username and password when prompted.

# N = number of securities sorted by market cap
# Exclude GICS codes

Features: active, date, gvkey,  year,  month,  mom1m,   mom3m,  mom6m,  mom9m,
        mrkcap, entval, saleq_ttm,      cogsq_ttm,      xsgaq_ttm,      oiadpq_ttm,
        niq_ttm,        cheq_mrq,       rectq_mrq,      invtq_mrq,      acoq_mrq,
        ppentq_mrq,     aoq_mrq,        dlcq_mrq,       apq_mrq,        txpq_mrq,
        lcoq_mrq,   ltq_mrq,    csho_1yr_avg

It takes around 30 mins to build the dataset for N=100 and date starting from 1980-01-01

"""

import wrds
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pickle
from time import time
from wrds_data_processing import DataProcessing
from configparser import SafeConfigParser, NoOptionError
import argparse as ap
import sys


start_time = time()

# Parse arguments
parser = ap.ArgumentParser(description="Build Data from WRDS")
parser.add_argument("--N",default=10,type=int,
                    help="Number of equities sorted by market cap")
parser.add_argument("--exclude_gics", default=[],
                    help="Excludes the industries with list of GICS codes")
parser.add_argument("--filename", help="Name of the output data file",
                    required = True)
parser.add_argument("--test_mode",default='no',help="Test mode with small N")

args = vars(parser.parse_args())
N = args['N']
try:
    exclude_gics = args['exclude_gics'].split(',')
except AttributeError:
    exclude_gics = args['exclude_gics']
out_filename = args['filename']
test_mode = args['test_mode']

# Connect to WRDS data engine
db = wrds.Connection()

#############################################################################
#### SQL Query-----------------------------------------------------------####
#############################################################################

# Initialize dictionary to store top N gvkeys for every month
top_gvkey_month = {}
top_N_eq_gvkey_list_all = set()

start_date = '2013-01-01'
curr_date = datetime.datetime.strptime(start_date,'%Y-%m-%d')

# Go through months starting with the start date and find top N companies by mrk cap
# for that month.

# Reference df for primary security
q10 = ("select gvkey,primiss from compm.secm")
primiss_df = db.raw_sql(q10)

while curr_date < datetime.datetime.now():
    # prev_date = curr_date + relativedelta(months=-3)
    curr_date_string = curr_date.strftime('%Y-%m-%d')
    # prev_date_string = prev_date.strftime('%Y-%m-%d')
    print(curr_date.date())

    # Query to get list of companies with top 2000 market cap for the given month
    q1a = ("select distinct a.gvkey,a.latest,b.cshoq,b.prccq,b.mkvaltq,b.cshoq*b.prccq as market_cap,b.curcdq "
         "from "
            "(select gvkey,max(datadate) as latest "
             "from "
             "compm.fundq where datadate < '%s' "
             "group by gvkey) a inner join "
                 "(select gvkey,datadate,mkvaltq,cshoq,prccq,curcdq "
                    "from compm.fundq where cshoq>0 and prccq>0 and curcdq='USD' and mkvaltq>0) b "
        "on a.gvkey = b.gvkey and a.latest=b.datadate "
         "order by market_cap desc "
        "limit %i")%(curr_date_string,N)

    mrk_df = db.raw_sql(q1a)
    # merge the security flag
    mrk_df = mrk_df.merge(primiss_df,on='gvkey',how='left')
    gvkey_list_month = mrk_df['gvkey'][mrk_df['primiss']=='P'].values.tolist()
    top_gvkey_month[curr_date.date()] = gvkey_list_month
    top_N_eq_gvkey_list_all |= set(gvkey_list_month)

    # increment the date for next month
    curr_date = curr_date + relativedelta(months=1)

top_N_eq_gvkey_list_all = list(top_N_eq_gvkey_list_all)

# Query to get GIC codes and remove the exclude_gics list
q1b = ("select gvkey,gsector "
     "from compa.company ")
df_gic = db.raw_sql(q1b)
exclude_gvkey_list = df_gic['gvkey'][df_gic['gsector'].isin(exclude_gics)].tolist()

# remove gvkey of associated gic code from the main list
top_N_eq_gvkey_list = [k for k in top_N_eq_gvkey_list_all if k not in exclude_gvkey_list]

# Check for continuation of companies and update their status (active or not)
# Compare the gvkey list with the most recent list if it exists
# Update the current gvkey list with the inactive ones

# Read the gvkey config file which contains the most recent list
config_gvkey = SafeConfigParser()
config_gvkey.read('gvkey-hist.dat')
config_gvkey.set('gvkey_list', '# Used to keep track of most recent requity list. No need to edit', '')

# Initialize active dict
active = {key: 1 for key in top_N_eq_gvkey_list}

if test_mode != 'yes':
    try:
        mr_gvk_list = config_gvkey.get('gvkey_list', 'most_recent_list').split(',')
        inactive_list = [k for k in mr_gvk_list if k not in top_N_eq_gvkey_list]

        # Add inactive gvkey
        for inactive_gvk in inactive_list:
            active[inactive_gvk] = 0

        # Update the current gvkey list with the inactive ones
        top_N_eq_gvkey_list = list(set().union(top_N_eq_gvkey_list,inactive_list))

        # create the most recent list in the config file if it doesn't exist
        config_gvkey.set('gvkey_list', 'most_recent_list', ','.join(top_N_eq_gvkey_list))

    except NoOptionError:
        # create the most recent list in the config file if it doesn't exist
        config_gvkey.set('gvkey_list', 'most_recent_list', ','.join(top_N_eq_gvkey_list))


# save to a file
with open('gvkey-hist.dat', 'w') as configfile:
    config_gvkey.write(configfile)

# change format to be compatible with sql query
top_N_eq_gvkey = tuple(["'%s'"%str(i) for i in top_N_eq_gvkey_list])
top_N_eq_gvkey = ",".join(top_N_eq_gvkey)

# Query to get fundamental Data
q2 = ("select datadate,gvkey,tic,saleq,cogsq,xsgaq,oiadpq,niq,"
      "cheq, rectq, invtq, acoq, ppentq, aoq, dlcq, apq, txpq, lcoq, ltq, dlttq, cshoq, seqq, atq "
    "from compm.fundq "
     "where gvkey in (%s) and datadate > '%s' ")%(top_N_eq_gvkey,start_date)
fundq_df = db.raw_sql(q2)

# Add gics_sector as a column
fundq_df = pd.merge(fundq_df,df_gic,how='left',on=['gvkey'])

# Query to get price data
q3 = ("select gvkey,datadate,prccm,ajexm "
     "from compm.secm "
     "where gvkey in (%s) ")%top_N_eq_gvkey
price_df_all = db.raw_sql(q3).sort_values('datadate')
price_df_all.datadate = pd.to_datetime(price_df_all.datadate,format='%Y-%m-%d')

# Query to get stock_split data
q4 = ("select gvkey,datadate,split "
     "from compm.sec_split "
     "where gvkey in (%s) ")%top_N_eq_gvkey
stock_split_df_all = db.raw_sql(q4).sort_values('datadate')
stock_split_df_all.datadate = pd.to_datetime(stock_split_df_all.datadate,format='%Y-%m-%d')

####--END OF SQL QUERYING-------------------------------------------------------

# Build balance sheet features
blnc_sheet_list = ['cheq','rectq','invtq','acoq','ppentq','aoq',
                                'dlcq','apq','txpq','lcoq','ltq','dlttq','cshoq','seqq','atq']

# Build income sheet features
income_list = ['saleq','cogsq','xsgaq','oiadpq','niq']

gvkey_list = top_N_eq_gvkey_list
print("Total Number of Equities in the dataset: %i"%len(gvkey_list))
print('\n')

df_all = fundq_df[['gvkey','gsector','datadate'] + income_list + blnc_sheet_list]
df_all['active'] = np.nan


def reorder_cols():
    a = ['active','datadate','gvkey','gsector','year','month']
    mom = ['mom1m','mom3m','mom6m','mom9m']
    prc = ['mrkcap','entval']
    ttm_list_tmp = [x + '_ttm' for x in income_list]
    mrq_list_tmp = [x + '_mrq' for x in blnc_sheet_list]
    mrq_list_tmp.remove('cshoq_mrq')
    mrq_list_tmp.remove('dlttq_mrq')
    csho = ['csho_1yr_avg']
    price = ['adjusted_price','prccm','ajexm']

    new_order = a + mom + prc + ttm_list_tmp + mrq_list_tmp + csho + price
    return new_order


# Create empty df to be appended for each equity
df_all_eq = pd.DataFrame(columns=reorder_cols())


# Start filling data by gvkey
for jj,key in enumerate(gvkey_list):
    try:
        t0=time()
        # print("GVKEY: %s"%key)
        df = df_all[df_all['gvkey'] == key].copy()
        df = df.sort_values('datadate')
        df = df.set_index('datadate',drop=False)
        df = df[~df.index.duplicated(keep='first')]
        # print("df shape:%g,%g"%df.shape)

        # get price_df for the current gvkey
        price_df = price_df_all[price_df_all['gvkey']==key].copy()
        # print("price df shape:%g,%g"%price_df.shape)

        # get stock_split_df for the current gvkey
        stock_split_df = stock_split_df_all[stock_split_df_all['gvkey']==key].copy()
        # print("stock split df shape:%g,%g"%stock_split_df.shape)
        # print("\n")

        # Start data processing
        dp = DataProcessing(lag=3, monthly_active_gvkey=top_gvkey_month)

        # Add the lag to the date index
        df = dp.add_lag(df)

        # Create new df with monthly frequency (empty)
        new_df_empty = dp.create_df_monthly(df)

        # Add ttm and mrq data
        ttm_mrq_df = dp.create_ttm_mrq(df, new_df_empty)

        # Add price information
        df_w_price, price_df_for_mom = dp.add_price_features(ttm_mrq_df, price_df)

        # Add momentum features
        df_w_mom = dp.get_mom(df_w_price, price_df_for_mom, [1, 3, 6, 9])

        # Add csho_1_year average
        df_w_mom['csho_1yr_avg'] = df_w_mom['cshoq_mrq'].rolling(12, min_periods=1).mean()

        # Reorder column names
        new_order = reorder_cols()

        del df, price_df, stock_split_df

        df_out = df_w_mom[new_order]

        # Fill Nans with 0.0
        df_out = df_out.fillna(0.0)
        df_out = df_out.reset_index(drop=True)

        # Append the current df to the full_df
        df_all_eq = df_all_eq.append(df_out, ignore_index=True)

        print("%i GVKEY: %s, Time %2.2f"%(jj, key, time()-t0))

    except (ValueError, IndexError):
        pass

# Normalize the momentum features
dates = df_all_eq['datadate'].unique()

mom_f = ['mom1m', 'mom3m', 'mom6m', 'mom9m']

for date in dates:
    date = pd.Timestamp(date)
    df_date = df_all_eq[mom_f][df_all_eq['datadate'] == date]

    ix_dates = df_date.index
    df_norm = (df_date - df_date.min())/(df_date.max() - df_date.min())

    df_norm = df_norm.fillna(0.0)

    df_all_eq.loc[ix_dates, mom_f] = df_norm

    del df_date, df_norm

# Change date label from 'datadate' to 'date'
df_all_eq.rename(columns={'datadate':'date'}, inplace=True)

# Change date from Y-m-d to ymd
df_all_eq['date'] = df_all_eq['date'].dt.strftime('%Y%m')

# Change column name gsector to gics-sector
df_all_eq = df_all_eq.rename(columns={'gsector':'gics-sector'})

# Output the csv
df_all_eq.to_csv(out_filename, sep=' ', index=False)

print('\n')
print("Shape of dataframe: %g,%g"%df_all_eq.shape)
print('\n')

exec_time = time() -start_time

print("Total Execution Time: %2.2f"%exec_time)
