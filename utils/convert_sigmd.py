# -*- coding: utf-8 -*-
##
# convert signal. normalized by sigma, then sigmoid [0, 1]
# return Y is time series matrix (node x time series)
# input:
#  X          multivariate time series matrix (node x time series)
#  centroid   signal centroid value for normalization (i.e. BOLD: set 0, other:auto)(option)

from __future__ import print_function, division

import math
import numpy as np
import scipy


class SigmoidConverter(object):
    def __init__(self):
        self

    def to_sigmoid_signal(self, x, centroid=float('NaN')):
        max_si = np.max(x)
        min_si = np.min(x)
        sig = math.sqrt(np.var(x, ddof=0,))
        if math.isnan(centroid):
            c = np.mean(x)
        else:
            c = centroid
        xn = (x - c) / sig
        y = scipy.special.expit(xn)
        return y, sig, c, max_si, min_si

    def inv_sigmoid_signal(self, y, sig, c, max_si, min_si):
        ya = y / (1 - y)
        y2 = np.log(ya)
        x = sig * y2 + c
        # replace inf to original max or min values
        x = np.where(np.isinf(x) & (x > 0), max_si, x)
        x = np.where(np.isinf(x) & (x < 0), min_si, x)
        return x

