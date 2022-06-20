# -*- coding: utf-8 -*-
##
# Estimate hidden neurons and initial weight and create multivariate VAR DNN
# input:
#  x               multivariate time series matrix (node x time series)
#  ex_signal       multivariate time series matrix (exogenous input x time series) (default:[])
#  node_control    node control matrix (node x node) (default:[])
#  ex_control      exogenous input control matrix for each node (node x exogenous input) (default:[])
#  lags            number of lags for autoregression (default:1)
#  reg_l2          L2 Regularization value (default:0.05)

from __future__ import print_function, division

import os
import math
import numpy as np
from keras.models import load_model
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam, Adagrad
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras import regularizers, initializers
import pickle


class MultivariateVARDNNetwork(object):
    def __init__(self):
        self.node_num = 0
        self.sig_len = 0
        self.ex_num = 0
        self.node_max = 0
        self.lags = 0
        self.models = []
        self.residuals = []

    def init(self, x, ex_signal=[], node_control=[], ex_control=[], lags=1, reg_l2=0.05):
        self.node_num = x.shape[0]
        self.sig_len = x.shape[1]
        self.lags = lags
        if len(ex_signal):
            self.ex_num = ex_signal.shape[0]
            x = np.concatenate([x, ex_signal], 0)
        else:
            self.ex_num = 0
        self.node_max = self.node_num + self.ex_num

        control = np.ones((self.node_num, lags*self.node_max))
        if len(node_control) == 0:
            node_control = np.ones((self.node_num, self.node_num))
        if len(ex_control) == 0:
            ex_control = np.ones((self.node_num, self.ex_num))
        for p in range(lags):
            control[:, self.node_max*p:self.node_max*(p+1)] = np.concatenate([node_control, ex_control], 1)

        input_nums = math.ceil((np.sum(control)) / self.node_num)
        hidden_nums = self.estimate_hidden_neurons(node_num=input_nums, sig_len=self.sig_len)
        reg_l2 = regularizers.L2(reg_l2)
        ini_randuni = initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)

        # create VARDNN networks
        for i in range(self.node_num):
            input_num = np.sum(control[i, :])
            model = Sequential()
            model.add(Dense(hidden_nums[0], input_shape=(int(input_num),), activation='relu',
                            kernel_initializer=ini_randuni, bias_initializer='zeros'))
            for j in range(1, len(hidden_nums)):
                model.add(Dense(hidden_nums[j], activation='relu', activity_regularizer=reg_l2,
                                kernel_initializer=ini_randuni, bias_initializer='zeros'))
            model.add(Dense(1, activation='linear',
                            kernel_initializer=ini_randuni, bias_initializer='zeros'))
            model.build()
#            model.summary()  # can be commented out
            adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
            model.compile(loss='mean_squared_error', optimizer=adam)
            self.models.append(model)


    def fit(self, x, ex_signal=[], node_control=[], ex_control=[], batch_size=100, epochs=500):
        if len(ex_signal):
            x = np.concatenate([x, ex_signal], 0)

        x = x.transpose()
        y = np.flipud(x)
        yt = np.zeros((self.sig_len-self.lags, self.lags*self.node_max))
        control = np.ones((self.node_num, self.lags*self.node_max))
        if len(node_control) == 0:
            node_control = np.ones((self.node_num, self.node_num))
        if len(ex_control) == 0:
            ex_control = np.ones((self.node_num, self.ex_num))
        for p in range(self.lags):
            yt[:, self.node_max*p:self.node_max*(p+1)] = y[1+p:self.sig_len-self.lags+1+p, :]
            control[:, self.node_max*p:self.node_max*(p+1)] = np.concatenate([node_control, ex_control], 1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)

        for i in range(self.node_num):
            idx = np.where(control[i, :] == 1)
            yi = y[0:self.sig_len - self.lags, i]
            xti = yt[:, idx[0]]
            print('training node ' + str(i))

            hist = self.models[i].fit(xti, yi,
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=0,
                              validation_split=0.2,
                              shuffle=True,
                              callbacks=[early_stopping])
            pred = self.models[i].predict(xti, verbose=0)
            r = (yi - pred)
            self.residuals.append(r)


    def estimate_hidden_neurons(self, node_num, sig_len, max_neuron_num=[]):
        hidden_nums = []
        num = math.ceil(32 + (sig_len-100)*0.12/(1+node_num*0.01))
        if len(max_neuron_num) and num > max_neuron_num:
            num = max_neuron_num
        hidden_nums.append(int(num))
        num = math.ceil(hidden_nums[0] * 2/3)
        hidden_nums.append(int(num))
        return hidden_nums

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
        self.models = []
        for i in range(self.node_num):
            model_file = path_name + os.sep + 'model' + str(i) + '.h5'
            model = load_model(model_file)
            self.models.append(model)

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
        for i in range(len(self.models)):
            model_file = path_name + os.sep + 'model' + str(i) + '.h5'
            self.models[i].save(model_file)

