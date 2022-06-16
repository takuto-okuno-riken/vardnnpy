# -*- coding: utf-8 -*-

from __future__ import print_function, division
import os
import datetime

import numpy as np
import scipy.io as sio
from measures.functional_connectivity import FunctionalConnectivity
from measures.partial_correlation import PartialCorrelation

class TestPartialCorrelation(object):
    def __init__(self):
        self.f_name = 'testTrain-rand500-uniform.mat'
        self.work_path = 'data'

    def test(self):
        data_file = os.path.join(self.work_path, self.f_name)
        fdata = sio.loadmat(data_file)
        node_num = 8   # node number
        sig_len = 200  # signal length
        si = np.array(fdata["si"])
        x = si[0:node_num, 0:sig_len]

        # set signal time lag 6->2, 6->4
        x[1, 1:sig_len] = x[5, 0:sig_len-1]
        x[3, 1:sig_len] = x[5, 0:sig_len-1]

        fc = FunctionalConnectivity()
        fc.plot(x=x)
        pc = PartialCorrelation()
        pc.plot(x=x)

        # set ex_signal time lag 1->3
        ex = si[node_num:node_num+2, 0:sig_len]
        x[2, 1:sig_len] = ex[0, 0:sig_len-1]
        fc.plot(x=x, ex_signal=ex, is_full_node=1)
        pc.plot(x=x, ex_signal=ex, is_full_node=1)

        # set node_control and ex_control
        node_control = np.ones((node_num, node_num))
        node_control[3, 3] = 0
        node_control[2, 5] = 0
        ex_control = np.ones((node_num, 2))
        ex_control[2, 1] = 0
        fc.plot(x=x, ex_signal=ex, node_control=node_control, ex_control=ex_control, is_full_node=1)
        pc.plot(x=x, ex_signal=ex, node_control=node_control, ex_control=ex_control, is_full_node=1)


if __name__ == '__main__':
    print('start partial correlation test')
    start_time = datetime.datetime.now()
    test_pc = TestPartialCorrelation()
    test_pc.test()
    end_time = datetime.datetime.now()
    interval = (end_time - start_time).seconds
    print('run time: %d seconds' % int(interval))
