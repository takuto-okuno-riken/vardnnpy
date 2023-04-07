# -*- coding: utf-8 -*-
##
# multivariate Vector Auto-Regression network class and Create mVAR network
# input:
#  x              multivariate time series matrix (node x time series)
#  ex_signal      multivariate time series matrix (exogenous input x time series) (optional)
#  node_control   node control matrix (node x node) (optional)
#  ex_control     exogenous input control matrix for each node (node x exogenous input) (optional)
#  lags           number of lags for autoregression (default:3)

from __future__ import print_function, division

import os
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression


class MultivariateVARNetwork(object):
    def __init__(self):
        self.node_num = 0
        self.sig_len = 0
        self.ex_num = 0
        self.node_max = 0
        self.lags = 0
        self.lr_objs = []
        self.residuals = []

    def init_with_matnet(self, dic):
        rvec = []
        bvec = []
        if type(dic) is np.ndarray:
            self.node_num = int(dic['nodeNum'][0, 0])
            self.sig_len = int(dic['sigLen'][0, 0])
            self.ex_num = int(dic['exNum'][0, 0])
            self.lags = int(dic['lags'][0, 0])
            self.cx_m = dic['cxM'][0, 0].flatten()
            self.cx_cov = dic['cxCov'][0, 0]
            bv = dic['bvec'][0, 0].flatten()
            rv = dic['rvec'][0, 0].flatten()
            for j in range(self.node_num):
                bvec.append(bv[j].flatten())
                rvec.append(rv[j].flatten())
        else:  # h5py
            self.node_num = int(dic['nodeNum'][0, 0])
            self.sig_len = int(dic['sigLen'][0, 0])
            self.ex_num = int(dic['exNum'][0, 0])
            self.lags = int(dic['lags'][0, 0])
            self.cx_m = dic['cxM'][::].flatten()
            self.cx_cov = dic['cxCov'][::]
            brefs = dic['bvec'][::].flatten()
            rrefs = dic['rvec'][::].flatten()
            for i in range(len(rrefs)):
                bvec.append(dic[brefs[i]][::].flatten())
                rvec.append(dic[rrefs[i]][::].flatten())
        self.node_max = self.node_num + self.ex_num
        for i in range(self.node_num):
            lr = LinearRegression(fit_intercept=True)
            b = bvec[i].flatten()
            lr.coef_ = b[0:len(b)-1]
            lr.intercept_ = b[len(b)-1]
            self.lr_objs.append(lr)
            self.residuals.append(rvec[i])

    def init(self, x, ex_signal=[], node_control=[], ex_control=[], lags=3):
        self.node_num = x.shape[0]
        self.sig_len = x.shape[1]
        self.lags = lags
        if len(ex_signal):
            self.ex_num = ex_signal.shape[0]
            x = np.concatenate([x, ex_signal], 0)
        else:
            self.ex_num = 0
        self.node_max = self.node_num + self.ex_num

        # set control
        control = np.ones((self.node_num, lags*self.node_max), dtype='bool')
        if len(node_control) == 0:
            node_control = np.ones((self.node_num, self.node_num), dtype='bool')
        if len(ex_control) == 0:
            ex_control = np.ones((self.node_num, self.ex_num), dtype='bool')

        x = x.transpose()
        y = np.flipud(x)
        yt = np.zeros((self.sig_len-lags, lags*self.node_max), dtype=x.dtype)
        for p in range(lags):
            yt[:, self.node_max*p:self.node_max*(p+1)] = y[1+p:self.sig_len-lags+1+p, :]
            control[:, self.node_max*p:self.node_max*(p+1)] = np.concatenate([node_control, ex_control], 1)

        for i in range(self.node_num):
            lr = LinearRegression(fit_intercept=True)
            idx = np.where(control[i, :] == 1)
            yi = y[0:self.sig_len - lags, i]
            xti = yt[:, idx[0]]

            lr.fit(xti, yi)
            pred = lr.predict(xti)
            r = (yi - pred)
            self.lr_objs.append(lr)
            self.residuals.append(r)

    def init_with_cell(self, cx, cex_signal=[], node_control=[], ex_control=[], lags=3):
        self.node_num = cx[0].shape[0]
        self.sig_len = cx[0].shape[1]
        self.lags = lags
        if len(cex_signal):
            self.ex_num = cex_signal[0].shape[0]
        else:
            self.ex_num = 0
        self.node_max = self.node_num + self.ex_num
        dtype = cx[0].dtype

        # set control
        control = np.ones((self.node_num, lags*self.node_max), dtype='bool')
        if len(node_control) == 0:
            node_control = np.ones((self.node_num, self.node_num), dtype='bool')
        if len(ex_control) == 0:
            ex_control = np.ones((self.node_num, self.ex_num), dtype='bool')
        for p in range(lags):
            control[:, self.node_max * p:self.node_max * (p + 1)] = np.concatenate([node_control, ex_control], 1)

        # calculate mean and covariance of each node
        y = cx[0]
        for i in range(1, len(cx)):
            y = np.concatenate([y, cx[i]], axis=1)
        self.cx_m = np.mean(y, axis=1, dtype=dtype)
        self.cx_cov = np.cov(y, dtype=dtype)
        del y  # clear memory

        all_in_len = 0
        for i in range(len(cx)):
            all_in_len += cx[i].shape[1] - lags

        # this implementation is memory consumption
        xts = 0
        xt = np.empty((all_in_len, self.node_num), dtype=dtype)  # smaller memory
        xti = np.empty((all_in_len, lags*self.node_max), dtype=dtype)  # smaller memory
        for i in range(len(cx)):
            x = cx[i]
            if len(cex_signal):
                x = np.concatenate([x, cex_signal[i]], 0)
            y = np.flipud(x.transpose())

            slen = y.shape[0]
            sl = slen - lags
            yt = np.empty((sl, lags*self.node_max), dtype=dtype)
            for p in range(lags):
                yt[:, self.node_max*p:self.node_max*(p+1)] = y[1+p:sl+1+p, :]

            xt[xts:xts+sl, :] = y[0:sl, 0:self.node_num]
            xti[xts:xts+sl, :] = yt[:, :]
            xts = xts + sl
            del x, y, yt

        for i in range(self.node_num):
            lr = LinearRegression(fit_intercept=True)
            idx = np.where(control[i, :] == 1)
            yi = xt[:, i]
            xti2 = xti[:, idx[0]]

            lr.fit(xti2, yi)
            pred = lr.predict(xti2)
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
        resi_file = path_name + os.sep + 'residuals.dat'
        with open(resi_file, 'rb') as p:
            self.residuals = pickle.load(p)
        reg_file = path_name + os.sep + 'regress.dat'
        with open(reg_file, 'rb') as p:
            self.lr_objs = pickle.load(p)

    def save(self, path_name):
        if not os.path.isdir(path_name):
            os.makedirs(path_name, exist_ok=True)
        list_file = path_name + os.sep + 'list.dat'
        dat = [self.node_num, self.sig_len, self.ex_num, self.node_max, self.lags]
        with open(list_file, 'wb') as p:
            pickle.dump(dat, p)
        resi_file = path_name + os.sep + 'residuals.dat'
        with open(resi_file, 'wb') as p:
            pickle.dump(self.residuals, p)
        reg_file = path_name + os.sep + 'regress.dat'
        with open(reg_file, 'wb') as p:
            pickle.dump(self.lr_objs, p)
