from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import datetime as dt
import numpy as np
from collections import OrderedDict


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
                    cur_output = lines[i + 6].split('  ')[self.output_field]
                    cur_target = lines[i + 7].split('  ')[self.output_field]

                    if cur_target == 'nan':
                        cur_target = 0.

                    cur_output = float(cur_output)
                    cur_target = float(cur_target)

                    gvkey = row[1]

                    try:
                        companies[gvkey][self.period]['output'][date] = cur_output
                        companies[gvkey][self.period]['target'][date] = cur_target
                    except KeyError:
                        companies[gvkey] = {}
                        companies[gvkey][self.period] = {}
                        companies[gvkey][self.period]['output'] = {}
                        companies[gvkey][self.period]['target'] = {}

                except (ValueError, IndexError):
                    pass

        return companies

    def get_errors(self, save_csv=False, err_filename='errors.csv'):
        """ Returns a dataframe of relative errors where rows are dates and columns are companies
            INPUTS
            companies: dict returned from read_predictions method
        """

        # Read the predictions files to generate company errors
        companies = self.read_predictions()

        # Initialize dict
        rel_err = {}

        print("Processing Errors...")

        for i, key in enumerate(sorted(companies)):
            # print(key)

            try:
                company = companies[key]
                p1 = company[1]

                out_p1 = sorted(p1['output'].items())
                tar_p1 = sorted(p1['target'].items())
                x1, y1 = zip(*out_p1)
                xt1, yt1 = zip(*tar_p1)

                rel_err[key] = abs(np.divide(np.array(y1) - np.array(yt1), np.array(yt1)))

                df_tmp = pd.DataFrame(data=rel_err[key], index=x1, columns=[key])
                df_tmp = df_tmp.replace([np.inf, -np.inf], np.nan)
                df_tmp = df_tmp.dropna()

                if i == 0:
                    df = df_tmp

                else:
                    df = pd.merge(df, df_tmp, how='outer', left_index=True, right_index=True)

            except ValueError:
                pass

            except:
                raise

            if i % 1000 == 0:
                print("%i companies processed" % i)
            #    break

        df = df.replace([np.inf, -np.inf], np.nan)

        print("Processing Complete")

        if save_csv:
            df.to_csv(err_filename)
            print("Error csv saved as %s" % err_filename)

        return df


if __name__ == '__main__':

    # Test
    train_file='test_EA/train-log-12.txt'
    pred_file='test_EA/predicts-rnn-fcst-pretty.dat'

    EA = ErrorAnalysis(train_file, pred_file)
    print("Reading train log")
    mse = EA.read_train_log()
    print(mse)
    print("getting errors")
    df_err = EA.get_errors(save_csv=True)
    print(df_err.head())