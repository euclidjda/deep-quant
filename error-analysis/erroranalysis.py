from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import datetime as dt
import numpy as np
from collections import OrderedDict
import os
import pickle

from errorplots import ErrorPlots


class ErrorAnalysis(object):
    """ Reads log and output files to analyze errors"""

    def __init__(self, train_log_file=None, pred_file=None, period=1, output_field=3):
        """ Instantiates the class with the log file and prediction output file

            period :        prediction period i.e how far out are the predictions in years (1,2,3 etc)
            output_field :  column to grab in the output file. EBIT is 3
        """

        self.train_log_file = train_log_file
        self.pred_file = pred_file
        self.period = period
        self.output_field = output_field

        return

    def read_train_log(self):
        """ Returns mse data from training log file
            mse is an ordered dict with epoch as key and (train_mse,validation_mse) as value
        """

        if self.train_log_file is None:
            print("train log file not provided")
            return

        mse_data = OrderedDict()

        # Iterate through the file
        with open(self.train_log_file) as f:
            lines = f.readlines()

            for line in lines:
                line = line.split(' ')
                if line[0] == 'Epoch:':
                    epoch = int(line[1])
                    train_mse = float(line[4])
                    valid_mse = float(line[7])

                    # Add to the mse dict
                    mse_data[epoch] = (train_mse, valid_mse)

        return mse_data

    def read_predictions(self):
        """ Returns a dict of companies with output and target values# Structure of companies dict
             companies : {
                         gvkey:
                             period: {
                                     output : { date: output }
                                     target : { date: target}
                                                             }
        """

        if self.pred_file is None:
            print('Predictions file not provided')
            return

        # initialize the dicts
        companies={}

        with open(self.pred_file, 'rb') as f:
            lines = f.readlines()

            for i, line in enumerate(lines):
                row = line.split(' ')
                try:
                    date = dt.datetime.strptime(str(row[0]), "%Y%m")
                    mse_val = float(row[-1].split('=')[-1])
                    cur_output = float(lines[i + 6].split('  ')[self.output_field])
                    cur_target = float(lines[i + 7].split('  ')[self.output_field])

                    if cur_target == 'nan':
                        cur_target = 0.

                    gvkey = row[1]

                    try:
                        companies[gvkey][self.period]['output'][date] = cur_output
                        companies[gvkey][self.period]['target'][date] = cur_target
                        companies[gvkey][self.period]['mse'][date] = mse_val
                    except KeyError:
                        companies[gvkey] = {}
                        companies[gvkey][self.period] = {}
                        companies[gvkey][self.period]['output'] = {}
                        companies[gvkey][self.period]['target'] = {}
                        companies[gvkey][self.period]['mse'] = {}

                except (ValueError, IndexError):
                    pass

        return companies

    def get_errors(self, save_csv=False, rel_err_filename='rel_error.csv',mse_err_filename='mse_error.csv'):
        """ Returns a dataframe of relative errors where rows are dates and columns are companies
            INPUTS
            companies: dict returned from read_predictions method
        """

        # Read the predictions files to generate company errors
        companies = self.read_predictions()
        pickle.dump(companies,open('companies.pkl','wb'))

        # Initialize dict
        rel_err = {}
        mse_err = {}

        print("Processing Errors...")

        for i, key in enumerate(sorted(companies)):
            # print(key)

            try:
                company = companies[key]
                p1 = company[1]

                out_p1 = sorted(p1['output'].items())
                tar_p1 = sorted(p1['target'].items())
                mse_p1 = sorted(p1['mse'].items())
                x1, y1 = zip(*out_p1)
                xt1, yt1 = zip(*tar_p1)
                x_mse_1,y_mse_1 = zip(*mse_p1)
                
                rel_err[key] = abs(np.divide(np.array(y1) - np.array(yt1), np.array(yt1)))
                mse_err[key] = np.array(y_mse_1)
                
                df_tmp = pd.DataFrame(data=rel_err[key], index=x1, columns=[key])
                df_tmp_mse = pd.DataFrame(data=mse_err[key], index=x1, columns=[key])
                df_tmp = df_tmp.replace([np.inf, -np.inf], np.nan)
                df_tmp_mse = df_tmp_mse.replace([np.inf, -np.inf], np.nan)
                df_tmp = df_tmp.dropna()
                df_tmp_mse = df_tmp_mse.dropna()

                if i == 0:
                    df = df_tmp
                    df_mse = df_tmp_mse

                else:
                    df = pd.merge(df, df_tmp, how='outer', left_index=True, right_index=True)
                    df_mse = pd.merge(df_mse, df_tmp_mse, how='outer', left_index=True, right_index=True)

            except ValueError:
                pass

            except:
                raise

            if i % 1000 == 0:
                print("%i companies processed" % i)
            #    break

        df = df.replace([np.inf, -np.inf], np.nan)
        df_mse = df_mse.replace([np.inf, -np.inf], np.nan)

        print("Processing Complete")

        if save_csv:
            df.to_csv(rel_err_filename)
            df_mse.to_csv(mse_err_filename)
            print("Relative Error csv saved as %s" % rel_err_filename)
            print("MSE Error csv saved as %s" % mse_err_filename)

        return df


if __name__ == '__main__':

    # Test
    pred_file = ("D:\\gcp_deep_cloud\\recreate-nips-2017-v3\\rnn\\predicts-rnn-pretty.dat")
    train_file = ("D:\\gcp_deep_cloud\\recreate-nips-2017-v3\\rnn\\output-rnn-train.txt")
    
    EA = ErrorAnalysis(train_file,pred_file)
    #EA = ErrorAnalysis(train_file)
    print("Reading train log")
    mse = EA.read_train_log()
    #print(mse)
    print("getting errors")
    df_err = EA.get_errors(save_csv=True)
    #df_err = pd.read_csv('errors.csv')
    print(df_err.head())

    #EP = ErrorPlots(mse, df_err)
    #EP = ErrorPlots(mse)
    #EP.plot_train_hist()
    #EP.plot_cdf(threshold=2.0)
    #print("Threshold count: %2.2f"%EP.get_threshold_count())
    #EP.plot_error_dist(threshold=2.0)