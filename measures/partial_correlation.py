# -*- coding: utf-8 -*-
##
# Calculate Partial correlation
# returns partial correlation matrix (pc_mat)
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


class PartialCorrelation(object):
    def __init__(self):
        self

    def calc(self, x, ex_signal=[], node_control=[], ex_control=[], is_full_node=0):
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
        pc_mat = np.zeros((node_num, node_max))
        pc_mat[:, :] = np.nan

        x = x.transpose()
        if len(node_control) == 0:
            node_control = np.ones((node_num, node_num))
        if len(ex_control) == 0:
            ex_control = np.ones((node_num, ex_num))
        control = np.concatenate([node_control, ex_control], 1)

        lr = LinearRegression(fit_intercept=True)
        for i in range(node_num):
            control2 = control[i, :].copy()
            control2[i] = 0
            xi = x[:, i]

            for j in range(i, node_max):
                if len(node_control) and j < node_num and node_control[i, j] == 0:
                    continue
                if len(ex_control) and j >= node_num and ex_control[i, j-node_num] == 0:
                    continue
                control3 = control2.copy()
                control3[j] = 0
                idx = np.where(control3 == 1)
                xtj = x[:, idx[0]]
                xj = x[:, j]

                lr.fit(xtj, xi)
                pred = lr.predict(xtj)
                r1 = (xi - pred)
                lr.fit(xtj, xj)
                pred = lr.predict(xtj)
                r2 = (xj - pred)

                pc_mat[i, j] = np.dot(r1.transpose(), r2) / (np.sqrt(np.dot(r1.transpose(), r1)) * np.sqrt(np.dot(r2.transpose(), r2)))
                if j < node_num:
                    pc_mat[j, i] = pc_mat[i, j]

        return pc_mat

    def plot(self, x, ex_signal=[], node_control=[], ex_control=[], is_full_node=0):
        pc_mat = self.calc(x=x, ex_signal=ex_signal, node_control=node_control, ex_control=ex_control, is_full_node=is_full_node)
        plt.matshow(pc_mat)
#        plt.axis('off')
        plt.title('Partial Correlation')
        plt.colorbar()
        plt.xlabel('Source Nodes')
        plt.ylabel('Target Nodes')
        plt.show(block=False)
        plt.pause(1)
        return pc_mat

