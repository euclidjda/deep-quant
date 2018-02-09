import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import sys

class data_processing(object):
    """ Process data from wrds into usable format """

    def __init__(self,lag = 3,equity_list=None):
        self.lag = lag
        self.equity_list = equity_list

        if self.lag%3 != 0:
            print("Enter the lag value in multples of 3. Lag frequency is in quarters")

        self.income_list = ['saleq','cogsq','xsgaq','oiadpq','niq']
        self.ttm_list = [x + '_ttm' for x in self.income_list]

        self.blnc_sheet_list = ['cheq','rectq','invtq','acoq','ppentq','aoq',
                                'dlcq','apq','txpq','lcoq','ltq','dlttq','cshoq']
        self.mrq_list = [x + '_mrq' for x in self.blnc_sheet_list]

    def add_1_day(self,date):
        """Adds 1 day to the given date"""
        return date + datetime.timedelta(days=1)

    def add_date_lag(self,date):
        """Adds the lag to the given date"""
        return date + relativedelta(months=self.lag)

    def add_lag(self,df):
        """Adds the lag to the index of the dataframe"""
        df.index = df.index.map(self.add_date_lag)
        return df

    def create_df_monthly(self,df):
        """Returns the new empty df with monthly frequency between start and
            end of orginal dataframe.

            Make sure index is date and data is date sorted
        """

        # define start and end period for monthly frequency
        # Date starts from the 1st of every month
        start = df.index[0] + datetime.timedelta(days=1)
        end = df.index[-1] + datetime.timedelta(days=1)
        date_range = pd.date_range(start,end,freq='MS')

        # create a new df with newly created date range
        new_df = pd.DataFrame(index=date_range)

        return new_df

    def get_mrq_date(self,date):
        """Returns last day of most recent quarter for which the financial
            data is present"""

        q_last_days = ((3,31),(6,30),(9,30),(12,31))

        m = date.month
        d = date.day
        y = date.year

        if (m,d) in q_last_days:
            mrq_date = date

        else:
            if m>=1 and m<=3:
                mrq_date = datetime.date(y-1,12,31)
            elif m>=4 and m<=6:
                mrq_date = datetime.date(y,3,31)
            elif m>=7 and m<=9:
                mrq_date = datetime.date(y,6,30)
            elif m>=10 and m<=12:
                mrq_date = datetime.date(y,9,30)

        return mrq_date

    def get_next_q_date(self,date):
        """Returns last day of most recent quarter for which the financial
            data is present"""

        q_last_days = ((3,31),(6,30),(9,30),(12,31))

        m = date.month
        d = date.day
        y = date.year

        if (m,d) in q_last_days:
            return date

        else:
            if m>=1 and m<=3:
                return datetime.date(y,3,31)
            elif m>=4 and m<=6:
                return datetime.date(y,6,30)
            elif m>=7 and m<=9:
                return datetime.date(y,9,30)
            elif m>=10 and m<=12:
                return datetime.date(y,12,31)


    def last_fin_year_range(self,date):
        """Returns the range of last financial year"""

        mrq = self.get_mrq_date(date)
        last_year_start = date - relativedelta(years=1)

        return [last_year_start,mrq]


    def fill_mrq(self,df,new_df):
        # Create a copy of df and change the index with 1 day
        df_next_day = df.copy()

        df_next_day.index = df_next_day.index.map(self.add_1_day)

        # Rename the columns of df_next_day to be same as new_df
        org_cols = self.income_list + self.blnc_sheet_list
        new_cols = self.ttm_list + self.mrq_list
        cols_pairing = dict(zip(org_cols,new_cols))
        df_next_day = df_next_day.rename(columns=cols_pairing)


        # Merge new_df with df_next_day
        # Using join operation is faster than filling mrq data within the loop
        new_df = pd.merge(new_df,df_next_day[self.mrq_list],how='left',
                                left_index=True,right_index=True)

        # Forward fill the MRQ data
        new_df[self.mrq_list]=new_df[self.mrq_list].fillna(method='ffill')
        new_df[self.mrq_list]=new_df[self.mrq_list].fillna(0.0)

        return new_df


    def create_ttm_mrq(self,df,new_df,status):
        """Returns the dataframe with _ttm and _mrq fields"""

        # Forward fill the origianl dataframe
        df = df.fillna(method='ffill')
        df = df.fillna(0.0)

        # Fill new_df with monthly frquency between 1st and last day of
        # orginal df
        cols = ['active','datadate','gvkey','year','month'] + self.ttm_list
        new_df = pd.concat([new_df,pd.DataFrame(columns=cols)])
        new_df = new_df[cols]

        # Update gvkey
        new_df['active'] = status
        new_df['datadate'] = new_df.index.values
        new_df['gvkey'] = df['gvkey'].iloc[0]
        new_df['year'] = [d.year for d in new_df.index]
        new_df['month'] = [d.month for d in new_df.index]

        # Fill the MRQ data
        new_df = self.fill_mrq(df,new_df)

        # Fill TTM data
        for i,date in enumerate(new_df.index):
            fin_year = self.last_fin_year_range(date.date())

            # Calculate the dates in df index that lie in the range
            z = [fin_year[0]<=j<=fin_year[1] for j in df.index]

            # Subset last4 quarter data
            last_4_q = df[self.income_list].loc[z]

            ttm = last_4_q.sum().to_frame().transpose()

            new_df.loc[date,self.ttm_list] = ttm[self.income_list].values[0]

        return new_df

    def adjust_cshoq(self,new_df,split_data):
        """Returns the new_df adjusted for stock split for number of shares
            outstanding"""

        for j,date in enumerate(split_data['datadate']):

            if (date.month,date.day) not in ((3,31),(6,30),(9,30),(12,31)):
                # Adjust with +1 day for the begining of the month
                next_q = self.get_next_q_date(date) +  datetime.timedelta(days=1)
                next_q_m = next_q.month
                split_date_list = [datetime.date(date.year,m,1) for m in range(date.month+1,next_q_m)]

                for split_date in split_date_list:
                    try:
                        new_df.loc[split_date,'cshoq_mrq'] = \
                        new_df.loc[split_date,'cshoq_mrq']*split_data['split'].iloc[j]
                    except KeyError:
                        pass

        return new_df

    def add_price_features(self,new_df,price_df):
        """Returns the new_df with market cap and enterprise value features.
            Also returns the price_df without date as index to be used in mom

        """

        # Adjust for date by adding 1 day. Price information is for the last
        # day of the month

        price_df.loc[:,'datadate'] = price_df['datadate'].apply(self.add_1_day)

        # Sort by date
        price_df = price_df.sort_values('datadate')

        # Fill the missing fields
        price_df = price_df.fillna(method='ffill')

        # Create a copy of price df without date as index to be used for mom calcs
        price_df_for_mom = price_df.copy()

        price_df = price_df.fillna(0.0)
        price_df = price_df.set_index('datadate',drop=False)
        # drop gvkey and datadate
        price_df = price_df.drop(['datadate','gvkey'],axis=1)
        price_df = price_df[~price_df.index.duplicated(keep='first')]

        new_df = pd.merge(new_df,price_df,how='left',left_index=True,
                            right_index=True)
        new_df = new_df.fillna(method='ffill')

        new_df['mrkcap'] = new_df['prccm']*new_df['cshoq_mrq']
        new_df['entval'] = new_df['mrkcap'] + new_df['dlttq_mrq'] + \
                            new_df['dlcq_mrq'] - new_df['cheq_mrq']

        return new_df,price_df_for_mom

    def get_mom(self,new_df,price_df,period):
        """Returns the new_df with the momentum features added.

            period is the list of back months for which momentum is calculated"""

        # NOte of filling missing values
        # price_df is filled with ffill and then 0.0 in the method "add_price_features"
        # To avoid zero divsion error, if the first element is NaN it is filled
        # 1e10 so that the division yields close to zero for momentum feature


        def period_price(row):
            """Returns the price p periods ago"""
            ix = row.name
            period_ix = ix - p

            try:
                return price_df['prccm'].loc[period_ix]
            except KeyError:
                return 1e10

        for p in period:
            price_df_p = price_df.apply(period_price,axis=1)
            price_df_p = price_df_p.fillna(1e10)
            price_df_copy = price_df.copy().fillna(0.0)

            feature_name = 'mom' + str(p) + 'm'
            mom_p = price_df_copy['prccm'].div(price_df_p)
            mom_p.index = pd.to_datetime(price_df_copy['datadate'].values)
            mom_p = mom_p.to_frame(name=feature_name)

            new_df = pd.merge(new_df,mom_p,how='left',left_index=True,
                                right_index=True)

            new_df = new_df.drop_duplicates(['datadate'])

            del mom_p

        return new_df

if __name__=='__main__':

    # Test
    y = pd.read_pickle('y.pkl')
    stock_split_df = pd.read_pickle('split.pkl')
    price_df = pd.read_pickle('prccm.pkl')

    dp = data_processing(y)
    df = y
    new_df_empty = dp.create_df_monthly(df)
    ttm_mrq_df = dp.create_ttm_mrq(df,new_df_empty)

    df_split_adjusted = dp.adjust_cshoq(ttm_mrq_df,stock_split_df)

    df_w_price,price_df_for_mom = dp.add_price_features(df_split_adjusted,price_df)

    df_w_mom = dp.get_mom(df_w_price,price_df_for_mom,[1,3,6,9])
