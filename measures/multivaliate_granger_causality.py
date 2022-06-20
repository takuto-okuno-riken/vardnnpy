# -*- coding: utf-8 -*-
##
# Calculate multivariate Granger Causality
# returns Granger causality index (gc_mat)
# input:
#  x              multivariate time series matrix (node x time series)
#  ex_signal      multivariate time series matrix (exogenous input x time series) (optional)
#  node_control   node control matrix (node x node) (optional)
#  ex_control     exogenous input control matrix for each node (node x exogenous input) (optional)
#  lags           number of lags for autoregression (default:3)
#  is_full_node   return both node & exogenous causality matrix (default:0)

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class MultivariateGrangerCausality(object):
    def __init__(self):
        self

    def calc(self, x, ex_signal=[], node_control=[], ex_control=[], lags=3, is_full_node=0):
        node_num = x.shape[0]
        sig_len = x.shape[1]
        if len(ex_signal):
            ex_num = ex_signal.shape[0]
            x = np.concatenate([x, ex_signal], 0)
        else:
            ex_num = 0

        if is_full_node != 0:
            node_max = node_num + ex_num
        else:
            node_max = node_num
        gc_mat = np.zeros((node_num, node_max))
        gc_mat[:, :] = np.nan

        x = x.transpose()
        y = np.flipud(x)
        yt = np.zeros((sig_len-lags, lags*node_max))
        control = np.ones((node_num, lags*node_max))
        if len(node_control) == 0:
            node_control = np.ones((node_num, node_num))
        if len(ex_control) == 0:
            ex_control = np.ones((node_num, ex_num))
        for p in range(lags):
            yt[:, node_max*p:node_max*(p+1)] = y[1+p:sig_len-lags+1+p, :]
            control[:, node_max*p:node_max*(p+1)] = np.concatenate([node_control, ex_control], 1)

        lr = LinearRegression(fit_intercept=True)
        for i in range(node_num):
            idx = np.where(control[i, :] == 1)
            yi = y[0:sig_len - lags, i]
            xti = yt[:, idx[0]]

            lr.fit(xti, yi)
            pred = lr.predict(xti)
            xr = (yi - pred)
            vit = np.var(xr, ddof=0)
            if vit == 0:
                vit = 1.0e-15  # avoid inf return

            for j in range(node_max):
                if i == j:
                    continue
                if len(node_control) and j < node_num and node_control[i, j] == 0:
                    continue
                if len(ex_control) and j >= node_num and ex_control[i, j-node_num] == 0:
                    continue

                control2 = control[i, :].copy()
                for p in range(lags):
                    control2[j + node_max * p] = 0
                idx = np.where(control2 == 1)
                xtj = yt[:, idx[0]]
                lr.fit(xtj, yi)
                pred = lr.predict(xtj)
                xr = (yi - pred)
                vjt = np.var(xr, ddof=0)

                gc_mat[i, j] = np.log(vjt / vit)

        return gc_mat

    def plot(self, x, ex_signal=[], node_control=[], ex_control=[], lags=3, is_full_node=0):
        gc_mat = self.calc(x=x, ex_signal=ex_signal, node_control=node_control, ex_control=ex_control, lags=lags, is_full_node=is_full_node)
        plt.matshow(gc_mat)
#        plt.axis('off')
        plt.title('Multivariate Granger Causality')
        plt.colorbar()
        plt.xlabel('Source Nodes')
        plt.ylabel('Target Nodes')
        plt.show(block=False)
        plt.pause(1)
        return gc_mat

