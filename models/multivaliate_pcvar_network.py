# -*- coding: utf-8 -*-
##
# multivariate Principle Component Vector Auto-Regression network class and Create mPCVAR network
# input:
#  x              multivariate time series matrix (node x time series)
#  ex_signal      multivariate time series matrix (exogenous input x time series) (optional)
#  node_control   node control matrix (node x node) (optional)
#  ex_control     exogenous input control matrix for each node (node x exogenous input) (optional)
#  lags           number of lags for autoregression (default:3)
#  explained_th   explained threshold for PCA components (default:0.99)

from __future__ import print_function, division

import os
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA


class MultivariatePCVARNetwork(object):
    def __init__(self):
        self.node_num = 0
        self.sig_len = 0
        self.ex_num = 0
        self.node_max = 0
        self.lags = 0
        self.explained_th = 0
        self.max_comps = []
        self.lr_objs = []
        self.pca_objs = []
        self.residuals = []

    def init(self, x, ex_signal=[], node_control=[], ex_control=[], lags=3, explained_th=0.99):
        self.node_num = x.shape[0]
        self.sig_len = x.shape[1]
        self.lags = lags
        self.explained_th = explained_th
        if len(ex_signal):
            self.ex_num = ex_signal.shape[0]
            x = np.concatenate([x, ex_signal], 0)
        else:
            self.ex_num = 0
        self.node_max = self.node_num + self.ex_num

        x = x.transpose()
        y = np.flipud(x)
        yt = np.zeros((self.sig_len-lags, lags*self.node_max))
        control = np.ones((self.node_num, lags*self.node_max))
        if len(node_control) == 0:
            node_control = np.ones((self.node_num, self.node_num))
        if len(ex_control) == 0:
            ex_control = np.ones((self.node_num, self.ex_num))
        for p in range(lags):
            yt[:, self.node_max*p:self.node_max*(p+1)] = y[1+p:self.sig_len-lags+1+p, :]
            control[:, self.node_max*p:self.node_max*(p+1)] = np.concatenate([node_control, ex_control], 1)

        for i in range(self.node_num):
            lr = LinearRegression(fit_intercept=True)
            pca = PCA()
            idx = np.where(control[i, :] == 1)
            yi = y[0:self.sig_len - lags, i]
            xti = yt[:, idx[0]]

            # pca
            pca.fit(xti)
            self.pca_objs.append(pca)
            feature = pca.transform(xti)

            max_comp = len(pca.explained_variance_ratio_)
            exp_total = 0
            for k in range(max_comp):
                exp_total = exp_total + pca.explained_variance_ratio_[k]
                if exp_total >= explained_th:
                    max_comp = k + 1
                    break
            self.max_comps.append(max_comp)

            # regression
            lr.fit(feature[:, 0:max_comp], yi)
            pred = lr.predict(feature[:, 0:max_comp])
            r = (yi - pred)
            self.lr_objs.append(lr)
            self.residuals.append(r)

    def load(self, path_name):
        list_file = path_name + os.sep + 'list.dat'
        with open(list_file, 'rb') as p:
            dat = pickle.load(p)
        self.node_num = dat[0]
        self.sig_len = dat[1]
        self.ex_num = dat[2]
        self.node_max = dat[3]
        self.lags = dat[4]
        self.explained_th = dat[5]
        self.max_comps = dat[6]
        resi_file = path_name + os.sep + 'residuals.dat'
        with open(resi_file, 'rb') as p:
            self.residuals = pickle.load(p)
        reg_file = path_name + os.sep + 'regress.dat'
        with open(reg_file, 'rb') as p:
            self.lr_objs = pickle.load(p)
        pca_file = path_name + os.sep + 'pca.dat'
        with open(pca_file, 'rb') as p:
            self.pca_objs = pickle.load(p)

    def save(self, path_name):
        if not os.path.isdir(path_name):
            os.makedirs(path_name, exist_ok=True)
        list_file = path_name + os.sep + 'list.dat'
        dat = [self.node_num, self.sig_len, self.ex_num, self.node_max, self.lags, self.explained_th, self.max_comps]
        with open(list_file, 'wb') as p:
            pickle.dump(dat, p)
        resi_file = path_name + os.sep + 'residuals.dat'
        with open(resi_file, 'wb') as p:
            pickle.dump(self.residuals, p)
        reg_file = path_name + os.sep + 'regress.dat'
        with open(reg_file, 'wb') as p:
            pickle.dump(self.lr_objs, p)
        pca_file = path_name + os.sep + 'pca.dat'
        with open(pca_file, 'wb') as p:
            pickle.dump(self.pca_objs, p)
