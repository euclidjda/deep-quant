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

Takes about 40 minutes to build the dataset for top 2000 equities and outputs a dat file
"""

import wrds
import pandas as pd
import datetime
import numpy as np
import pickle
from time import time
from wrds_data_processing import data_processing
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

# Query to get list of companies with top 2000 market cap
q1a = ("select a.gvkey,a.latest,b.cshoq,b.prccq,b.mkvaltq,b.cshoq*b.prccq as market_cap,b.curcdq "
     "from "
        "(select gvkey,max(datadate) as latest "
         "from "
         "compm.fundq where datadate > '2017-01-01' "
         "group by gvkey) a inner join "
             "(select gvkey,datadate,mkvaltq,cshoq,prccq,curcdq "
                "from compm.fundq where cshoq>0 and prccq>0 and curcdq='USD') b "
    "on a.gvkey = b.gvkey and a.latest=b.datadate "
     "order by market_cap desc "
    "limit %i")%N

mrk_df = db.raw_sql(q1a)
top_N_eq_gvkey_list_all = mrk_df['gvkey'].values.tolist()

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
config_gvkey.set('gvkey_list','# Used to keep track of most recent requity list. No need to edit','')

# Initialize active dict
active = {key:1 for key in top_N_eq_gvkey_list}

if test_mode != 'yes':
    try:
        mr_gvk_list = config_gvkey.get('gvkey_list','most_recent_list').split(',')
        inactive_list = [k for k in mr_gvk_list if k not in top_N_eq_gvkey_list]

        # Add inactive gvkey
        for inactive_gvk in inactive_list:
            active[inactive_gvk] = 0

        # Update the current gvkey list with the inactive ones
        top_N_eq_gvkey_list = list(set().union(top_N_eq_gvkey_list,inactive_list))

        # create the most recent list in the config file if it doesn't exist
        config_gvkey.set('gvkey_list','most_recent_list',','.join(top_N_eq_gvkey_list))

    except NoOptionError:
        # create the most recent list in the config file if it doesn't exist
        config_gvkey.set('gvkey_list','most_recent_list',','.join(top_N_eq_gvkey_list))


# save to a file
with open('gvkey-hist.dat', 'w') as configfile:
    config_gvkey.write(configfile)

# change format to be compatible with sql query
top_N_eq_gvkey = tuple(["'%s'"%str(i) for i in top_N_eq_gvkey_list])
top_N_eq_gvkey = ",".join(top_N_eq_gvkey)

# Query to get fundamental Data
q2 = ("select datadate,gvkey,tic,saleq,cogsq,xsgaq,oiadpq,niq,"
      "cheq, rectq, invtq, acoq, ppentq, aoq, dlcq, apq, txpq, lcoq, ltq, dlttq,cshoq "
    "from compm.fundq "
     "where gvkey in (%s) ")%top_N_eq_gvkey
fundq_df = db.raw_sql(q2)
print('\n')
print("Shape of raw dataframe: %g,%g"%fundq_df.shape)
print('\n')

# Query to get price data
q3 = ("select gvkey,datadate,prccm "
     "from compm.secm "
     "where gvkey in (%s) ")%top_N_eq_gvkey
price_df_all = db.raw_sql(q3).sort_values('datadate')

# Query to get stock_split data
q4 = ("select gvkey,datadate,split "
     "from compm.sec_split "
     "where gvkey in (%s) ")%top_N_eq_gvkey
stock_split_df_all = db.raw_sql(q4).sort_values('datadate')

####--END OF SQL QUERYING-------------------------------------------------------

# Build balance sheet features
blnc_sheet_list = ['cheq','rectq','invtq','acoq','ppentq','aoq',
                                'dlcq','apq','txpq','lcoq','ltq','dlttq','cshoq']

# Build income sheet features
income_list = ['saleq','cogsq','xsgaq','oiadpq','niq']

gvkey_list = top_N_eq_gvkey_list
print("Total Number of Equities in the dataset: %i"%len(gvkey_list))
print('\n')

df_all = fundq_df[['gvkey','datadate'] + income_list + blnc_sheet_list]
df_all['active'] = np.nan

def reorder_cols():
    a = ['active','datadate','gvkey','year','month']
    mom = ['mom1m','mom3m','mom6m','mom9m']
    prc = ['mrkcap','entval']
    ttm_list_tmp = [x + '_ttm' for x in income_list]
    mrq_list_tmp = [x + '_mrq' for x in blnc_sheet_list]
    mrq_list_tmp.remove('cshoq_mrq')
    mrq_list_tmp.remove('dlttq_mrq')
    csho = ['csho_1yr_avg']

    new_order = a + mom + prc + ttm_list_tmp + mrq_list_tmp + csho
    return new_order

# Create empty df to be appended for each equity
df_all_eq = pd.DataFrame(columns=reorder_cols())

# Start filling data by gvkey
for key in gvkey_list:
    #print("GVKEY: %s"%key)
    df = df_all[df_all['gvkey'] == key].copy()
    df = df.sort_values('datadate')
    df = df.set_index('datadate',drop=False)
    df = df[~df.index.duplicated(keep='first')]
    #print("df shape:%g,%g"%df.shape)

    # get price_df for the current gvkey
    price_df = price_df_all[price_df_all['gvkey']==key].copy()
    #print("price df shape:%g,%g"%price_df.shape)

    # get stock_split_df for the current gvkey
    stock_split_df = stock_split_df_all[stock_split_df_all['gvkey']==key].copy()
    #print("stock split df shape:%g,%g"%stock_split_df.shape)
    #print("\n")

    # Start data processing
    dp = data_processing(lag=3)

    # Add the lag to the date index
    df = dp.add_lag(df)

    # Create new df with monthly frequency (empty)
    new_df_empty = dp.create_df_monthly(df)

    # Add ttm and mrq data
    status = active[key]
    ttm_mrq_df = dp.create_ttm_mrq(df,new_df_empty,status)

    # Adjust for stock split
    df_split_adjusted = dp.adjust_cshoq(ttm_mrq_df,stock_split_df)

    # Add price information
    df_w_price,price_df_for_mom = dp.add_price_features(df_split_adjusted,price_df)

    # Add momentum features
    df_w_mom = dp.get_mom(df_w_price,price_df_for_mom,[1,3,6,9])

    # Add csho_1_year average
    df_w_mom['csho_1yr_avg'] = df_w_mom['cshoq_mrq'].rolling(12,min_periods=1).mean()

    # Reorder column names
    new_order = reorder_cols()

    del df,price_df,stock_split_df

    df_out = df_w_mom[new_order]

    # Fill Nans with 0.0
    df_out = df_out.fillna(0.0)
    df_out = df_out.reset_index(drop=True)

    # Append the current df to the full_df
    df_all_eq = df_all_eq.append(df_out,ignore_index=True)

# Normalize the momentum features
dates = df_all_eq['datadate'].unique()

mom_f = ['mom1m','mom3m','mom6m','mom9m']

for date in dates:
    date = pd.Timestamp(date)
    df_date = df_all_eq[mom_f][df_all_eq['datadate']==date]

    ix_dates = df_date.index
    df_norm = (df_date - df_date.min())/(df_date.max() - df_date.min())

    df_norm = df_norm.fillna(0.0)

    df_all_eq.loc[ix_dates,mom_f] = df_norm

    del df_date, df_norm

# Change date label from 'datadate' to 'date'
df_all_eq.rename(columns={'datadate':'date'},inplace=True)

# Change date from Y-m-d to ymd
df_all_eq['date'] = df_all_eq['date'].dt.strftime('%Y%m%d')

# Output the csv
df_all_eq.to_csv(out_filename,sep=' ',index=False)

exec_time = time() -start_time

print("Total Execution Time: %2.2f"%exec_time)
