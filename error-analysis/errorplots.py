from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style()


class ErrorPlots(object):
    """ Creates various plots to analyze training data and error plots"""

    def __init__(self, train_mse_data=None, err_df=None):
        """ Instantiates the class with training mse data and error df"""

        self.train_mse_data = train_mse_data
        self.err_df = err_df

    def plot_train_hist(self,filename=None):
        """ Plots the training and validation error history and saves with the given filename"""

        if not self.train_mse_data:
            print("MSE data not provided")
            return

        epoch = self.train_mse_data.keys()
        train_err = [x[0] for x in self.train_mse_data.values()]
        val_err = [x[1] for x in self.train_mse_data.values()]

        plt.plot(epoch, train_err, label='Train Error')
        plt.plot(epoch, val_err,'--',label='Val Error')
        plt.title("Training MSE History")
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.legend()
        if filename == None:
            plt.savefig('train_hist.png')
        else:
            plt.savefig(filename)
        plt.clf()
        return

    def plot_cdf(self, threshold=1.):
        """ Plots the cumulative distribution of mean percentage error"""

        if self.err_df is None:
            print("Error DF not provided")
            return

        mean = self.err_df.mean().dropna()
        sns.distplot(mean[mean < threshold], hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True))
        plt.title('CDF of Mean Percentage Error', fontsize=15)
        plt.xlabel("Mean Percentage Error", fontsize=15)
        plt.savefig("cdf.png")
        plt.clf()
        return

    def get_threshold_count(self, threshold=0.2):
        """ Returns the percentage of equities with mean percentage error below the threshold value"""

        mean = self.err_df.mean().dropna()
        total_comps = mean.count()
        perc_lt_thrshld = mean[mean < threshold].count() * 1.0 / total_comps
        return perc_lt_thrshld

    def plot_error_dist(self, threshold=0.2):
        """ Returns the error distribution of mean percentage errors"""

        mean = self.err_df.mean().dropna()
        sns.distplot(mean[mean < threshold])
        plt.title('Mean Percentage Error Distribution of Companies', fontsize=15)
        plt.xlabel('Mean Percentage Error', fontsize=15)
        plt.savefig("dist.png")
        plt.clf()
        return
