# -*- coding: utf-8 -*-
##
# Calculate multivariate VARDNN Granger Causality
# returns mVARDNN Granger causality index (gc_mat)
# input:
#  x              multivariate time series matrix (node x time series)
#  ex_signal      multivariate time series matrix (exogenous input x time series) (optional)
#  node_control   node control matrix (node x node) (optional)
#  ex_control     exogenous input control matrix for each node (node x exogenous input) (optional)
#  net            mVARDNN network instance
#  is_full_node   return both node & exogenous causality matrix (default:0)

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt


class MultivariateVARDNNGrangerCausality(object):
    def __init__(self):
        self

    def calc(self, x, ex_signal=[], node_control=[], ex_control=[], net=[], is_full_node=0):
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
        yt = np.zeros((sig_len-net.lags, net.lags*node_max))
        control = np.ones((node_num, net.lags*node_max))
        if len(node_control) == 0:
            node_control = np.ones((node_num, node_num))
        if len(ex_control) == 0:
            ex_control = np.ones((node_num, ex_num))
        for p in range(net.lags):
            yt[:, node_max*p:node_max*(p+1)] = y[1+p:sig_len-net.lags+1+p, :]
            control[:, node_max*p:node_max*(p+1)] = np.concatenate([node_control, ex_control], 1)

        for i in range(node_num):
            idx = np.where(control[i, :] == 1)
            yi = y[0:sig_len - net.lags, i]

            r = net.residuals[i]
            vit = np.var(r, ddof=0)
            if vit == 0:
                vit = 1.0e-15  # avoid inf return

            for j in range(node_max):
                if i == j:
                    continue

                xtj = yt.copy()
                for p in range(net.lags):
                    xtj[:, j + node_max * p] = 0
                xtj = xtj[:, idx[0]]

                pred = net.models[i].predict(xtj, verbose=0)
                xr = (yi - pred)
                vjt = np.var(xr, ddof=0)

                gc_mat[i, j] = np.log(vit / vjt)  # shouldn't it be opposite?

        return gc_mat

    def plot(self, x, ex_signal=[], node_control=[], ex_control=[], net=[], is_full_node=0):
        gc_mat = self.calc(x=x, ex_signal=ex_signal, node_control=node_control, ex_control=ex_control,
                           net=net, is_full_node=is_full_node)
        plt.matshow(gc_mat)
#        plt.axis('off')
        plt.title('Multivariate VARDNN Granger Causality')
        plt.colorbar()
        plt.xlabel('Source Nodes')
        plt.ylabel('Target Nodes')
        plt.show()
        return gc_mat

