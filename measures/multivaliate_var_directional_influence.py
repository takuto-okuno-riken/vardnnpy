# -*- coding: utf-8 -*-
##
# Calculate mVAR (multivaliate Vector Auto-Regression) DI
# returns mVAR DI matrix (DI)
# input:
#  net            mVAR network instance
#  node_control   node control matrix (node x node) (optional)
#  ex_control     exogenous input control matrix for each node (node x exogenous input) (optional)
#  is_full_node   return both node & exogenous causality matrix (default:0)

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class MultivariateVARDirectionalInfluence(object):
    def __init__(self):
        self

    def calc(self, net=[], node_control=[], ex_control=[], is_full_node=0):
        node_num = net.node_num
        sig_len = net.sig_len
        ex_num = net.ex_num
        lags = net.lags

        if is_full_node != 0:
            node_max = node_num + ex_num
        else:
            node_max = node_num
        di_mat = np.zeros((node_num, node_max))
        di_mat[:, :] = np.nan

        control = np.ones((node_num, lags*node_max))
        if len(node_control) == 0:
            node_control = np.ones((node_num, node_num))
        if len(ex_control) == 0:
            ex_control = np.ones((node_num, ex_num))
        for p in range(lags):
            control[:, node_max*p:node_max*(p+1)] = np.concatenate([node_control, ex_control], 1)

        one_input = np.ones((1, lags*node_max))
        for i in range(node_num):
            idx = np.where(control[i, :] == 1)
            xti = one_input[:, idx[0]]

            z = net.lr_objs[i].predict(xti)

            for j in range(node_max):
                if i == j:
                    continue
                control2 = control[i, :].copy()
                for p in range(lags):
                    control2[j + node_max * p] = 2
                xti[:, :] = control2[idx[0]]
                xtj = np.where(xti == 2, 0, xti)
                zj = net.lr_objs[i].predict(xtj)

                di_mat[i, j] = np.abs(z - zj)

        return di_mat

    def plot(self, net=[], node_control=[], ex_control=[], is_full_node=0):
        di_mat = self.calc(net=net, node_control=node_control, ex_control=ex_control, is_full_node=is_full_node)
        plt.matshow(di_mat)
#        plt.axis('off')
        plt.title('Multivariate VAR Directional Influence')
        plt.colorbar()
        plt.xlabel('Source Nodes')
        plt.ylabel('Target Nodes')
        plt.show()
        return di_mat

