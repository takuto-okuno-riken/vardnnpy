# -*- coding: utf-8 -*-
##
# Calculate Functional Connectivity
# returns Functional Connectivity (fc_mat)
# input:
#  X              multivariate time series matrix (node x time series)
#  ex_signal      multivariate time series matrix (exogenous input x time series) (optional)
#  node_control   node control matrix (node x node) (optional)
#  ex_control     exogenous input control matrix for each node (node x exogenous input) (optional)
#  is_full_node   return both node & exogenous matrix (optional)

from __future__ import print_function, division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class FunctionalConnectivity(object):
    def __init__(self):
        self

    def calc(self, x, ex_signal=[], node_control=[], ex_control=[], is_full_node=0):
        node_num = x.shape[0]
        if len(ex_signal):
            ex_num = ex_signal.shape[0]
            x = np.concatenate([x, ex_signal], 0)
        else:
            ex_num = 0

        if is_full_node != 0:
            node_max = node_num + ex_num
        else:
            node_max = node_num
        fc_mat = np.zeros((node_num, node_max))
        fc_mat[:, :] = np.nan

        for i in range(node_num):
            xi = x[i, :]
            si = pd.Series(xi)
            for j in range(i, node_max):
                if len(node_control) and j < node_num and node_control[i, j] == 0:
                    continue
                if len(ex_control) and j >= node_num and ex_control[i, j-node_num] == 0:
                    continue
                xj = x[j, :]
                sj = pd.Series(xj)
                fc_mat[i, j] = si.corr(sj)
                if j < node_num:
                    fc_mat[j, i] = fc_mat[i, j]

        return fc_mat

    def plot(self, x, ex_signal=[], node_control=[], ex_control=[], is_full_node=0):
        fc_mat = self.calc(x=x, ex_signal=ex_signal, node_control=node_control, ex_control=ex_control, is_full_node=is_full_node)
        plt.matshow(fc_mat, vmin=-1, vmax=1)
#        plt.axis('off')
        plt.title('Functional Connectivity')
        plt.colorbar()
#        ax = plt.gca()
        plt.xlabel('Source Nodes')
        plt.ylabel('Target Nodes')
        plt.show()
        return fc_mat

