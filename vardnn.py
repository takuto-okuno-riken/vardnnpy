# -*- coding: utf-8 -*-

from __future__ import print_function, division

import sys

import numpy as np
import scipy.io as sio
import os
import pandas as pd
import matplotlib.pyplot as plt

from utils.parse_options import ParseOptions
from utils.convert_sigmd import SigmoidConverter
import measures
import models

# -------------------------------------------------------------------------
# matrix calculation


def get_optional_signals(Cex, Cctrl, Cexctrl, i):
    if len(Cex) == 1:
        ex_signal = Cex[0]
        is_full_node = 1
    elif len(Cex) > i:
        ex_signal = Cex[i]
        is_full_node = 1
    else:
        ex_signal = []
        is_full_node = 0

    if len(Cctrl) == 1:
        node_control = Cctrl[0]
    elif len(Cctrl) > i:
        node_control = Cctrl[i]
    else:
        node_control = []

    if len(Cexctrl) == 1:
        ex_control = Cexctrl[0]
    elif len(Cexctrl) > i:
        ex_control = Cexctrl[i]
    else:
        ex_control = []
    return ex_signal, node_control, ex_control, is_full_node

def save_mat_file(opt, mat, CXnames, algoname):
    out_path = opt.outpath + os.sep
    if opt.format == 0:  # csv each
        for i in range(len(mat)):
            f_name = out_path + CXnames[i] + '_' + algoname + '.csv'
            print('saving matrix : ' + f_name)
            np.savetxt(f_name, mat[i], delimiter=',')

    elif opt.format == 1:  # mat each
        for i in range(len(mat)):
            f_name = out_path + CXnames[i] + '_' + algoname + '.mat'
            print('saving matrix : ' + f_name)
            sio.savemat(f_name, {'Index': mat[i]})

    elif opt.format == 2:  # mat all
        f_name = out_path + CXnames[0] + '_' + algoname + '_all.mat'
        print('saving matrix : ' + f_name)
        # make 3D matrix
        mat3 = np.zeros((mat[0].shape[0], mat[0].shape[1], len(mat)))
        for i in range(len(mat)):
            mat3[:, :, i] = mat[i]
        sio.savemat(f_name, {'Index': mat3})

# -------------------------------------------------------------------------
# main

if __name__ == '__main__':
    options = ParseOptions()
    opt = options.parse()

    if type(opt.outpath) is list:
        opt.outpath = opt.outpath[0]  # replaced by string

    # read time-series and control files
    CXnames = []
    CX = []
    Cex = []
    Cctrl = []
    Cexctrl = []
    for i in range(len(opt.in_files)):
        if not os.path.isfile(opt.in_files[i]):
            print('bad file name. ignore : ' + opt.in_files[i])
            continue

        print('loading signals : ' + opt.in_files[i])
        name = os.path.splitext(os.path.basename(opt.in_files[i]))[0]
        if '.csv' in opt.in_files[i]:
            csv_input = pd.read_csv(opt.in_files[i], header=None)
            CX.append(csv_input.values)
            CXnames.append(name)
        elif '.mat' in opt.in_files[i]:
            dic = sio.loadmat(opt.in_files[i])
            X = dic.get('X')
            if X is not None and len(X) > 0:
                CX.append(X)
                CXnames.append(name)
            ex = dic.get('exSignal')
            if ex is not None and len(ex) > 0:
                Cex.append(ex)
            ctrl = dic.get('nodeControl')
            if ctrl is not None and len(ctrl) > 0:
                Cctrl.append(ctrl)
            ctrl = dic.get('exControl')
            if ctrl is not None and len(ctrl) > 0:
                Cexctrl.append(ctrl)

    if len(CX) == 0:
        print('no input files. exit script.')
        sys.exit()

    if opt.ex is not None:
        ex_files = opt.ex[0].split(':')
        for i in range(len(ex_files)):
            print('loading ex signals : ' + ex_files[i])
            if '.csv' in ex_files[i]:
                csv_input = pd.read_csv(filepath_or_buffer=ex_files[i], header=None)
                Cex.append(csv_input.values)

    # read control files
    if opt.nctrl is not None:
        nc_files = opt.nctrl[0].split(':')
        for i in range(len(nc_files)):
            print('loading node control : ' + nc_files[i])
            if '.csv' in nc_files[i]:
                csv_input = pd.read_csv(filepath_or_buffer=nc_files[i], header=None)
                Cctrl.append(csv_input.values)

    if opt.ectrl is not None:
        ec_files = opt.ectrl[0].split(':')
        for i in range(len(ec_files)):
            print('loading ex control : ' + ec_files[i])
            if '.csv' in ec_files[i]:
                csv_input = pd.read_csv(filepath_or_buffer=ec_files[i], header=None)
                Cexctrl.append(csv_input.values)

    # convert input & exogenous signals
    if opt.transform == 1:
        conv = SigmoidConverter()
        for i in range(len(CX)):
            CX[i], sig, c, max_si, min_si = conv.to_sigmoid_signal(x=CX[i], centroid=opt.transopt)
        for i in range(len(Cex)):
            Cex[i], sig, c, max_si, min_si = conv.to_sigmoid_signal(x=Cex[i], centroid=opt.transopt)


    # show input signals
    if opt.showsig:
        for i in range(len(CX)):
            plt.plot(CX[i].transpose(), linewidth=0.3)
            plt.title('Input time-series : ' + CXnames[i])
            plt.xlabel('Time frames')
            plt.show(block=False)

    if opt.showex:
        for i in range(len(Cex)):
            plt.plot(Cex[i].transpose(), linewidth=0.3)
            plt.title('Exogenous time-series : ' + CXnames[i])
            plt.xlabel('Time frames')
            plt.show(block=False)
    plt.pause(1)

    # -------------------------------------------------------------------------
    # matrix calculation

    # calc and plot FunctionalConnectivity
    fc_mat = []
    if opt.fc:
        for i in range(len(CX)):
            fc = measures.FunctionalConnectivity()
            ex_signal, node_control, ex_control, is_full_node = get_optional_signals(Cex, Cctrl, Cexctrl, i)

            if opt.showmat:
                mat = fc.plot(x=CX[i], ex_signal=ex_signal,
                              node_control=node_control, ex_control=ex_control, is_full_node=is_full_node)
            else:
                mat = fc.calc(x=CX[i], ex_signal=ex_signal,
                              node_control=node_control, ex_control=ex_control, is_full_node=is_full_node)
            fc_mat.append(mat)
        save_mat_file(opt, fc_mat, CXnames, 'fc')

    # calc and plot PartialCorrelation
    pc_mat = []
    if opt.pc:
        for i in range(len(CX)):
            pc = measures.PartialCorrelation()
            ex_signal, node_control, ex_control, is_full_node = get_optional_signals(Cex, Cctrl, Cexctrl, i)

            if opt.showmat:
                mat = pc.plot(x=CX[i], ex_signal=ex_signal,
                              node_control=node_control, ex_control=ex_control, is_full_node=is_full_node)
            else:
                mat = pc.calc(x=CX[i], ex_signal=ex_signal,
                              node_control=node_control, ex_control=ex_control, is_full_node=is_full_node)
            pc_mat.append(mat)
        save_mat_file(opt, pc_mat, CXnames, 'pc')

    # calc and plot pairwise Granger Causality
    pgc_mat = []
    if opt.pwgc:
        for i in range(len(CX)):
            pgc = measures.PairwiseGrangerCausality()
            ex_signal, node_control, ex_control, is_full_node = get_optional_signals(Cex, Cctrl, Cexctrl, i)

            if opt.showmat:
                mat = pgc.plot(x=CX[i], ex_signal=ex_signal, lags=opt.lag,
                              node_control=node_control, ex_control=ex_control, is_full_node=is_full_node)
            else:
                mat = pgc.calc(x=CX[i], ex_signal=ex_signal, lags=opt.lag,
                              node_control=node_control, ex_control=ex_control, is_full_node=is_full_node)
            pgc_mat.append(mat)
        save_mat_file(opt, pgc_mat, CXnames, 'pwgc')

    # calc and plot multivariate Granger Causality
    mgc_mat = []
    if opt.mvgc:
        for i in range(len(CX)):
            mgc = measures.MultivariateGrangerCausality()
            ex_signal, node_control, ex_control, is_full_node = get_optional_signals(Cex, Cctrl, Cexctrl, i)

            if opt.showmat:
                mat = mgc.plot(x=CX[i], ex_signal=ex_signal, lags=opt.lag,
                              node_control=node_control, ex_control=ex_control, is_full_node=is_full_node)
            else:
                mat = mgc.calc(x=CX[i], ex_signal=ex_signal, lags=opt.lag,
                              node_control=node_control, ex_control=ex_control, is_full_node=is_full_node)
            mgc_mat.append(mat)
        save_mat_file(opt, mgc_mat, CXnames, 'mvgc')


    # calc and plot VARDNN Directional Influence
    vdi_mat = []
    if opt.vddi:
        for i in range(len(CX)):
            cache_path = 'results' + os.sep + 'cache-vardnn-' + CXnames[i]
            net = models.MultivariateVARDNNetwork()
            if os.path.isdir(cache_path) and not opt.nocache:
                net.load(cache_path)
            else:
                ex_signal, node_control, ex_control, is_full_node = get_optional_signals(Cex, Cctrl, Cexctrl, i)
                net.init(x=CX[i], ex_signal=ex_signal, node_control=node_control, ex_control=ex_control,
                         lags=opt.lag, reg_l2=opt.l2)
                net.fit(x=CX[i], ex_signal=ex_signal, node_control=node_control, ex_control=ex_control,
                        epochs=opt.epoch, batch_size=int(CX[i].shape[1]/3))
                if not opt.nocache:
                    net.save(cache_path)

            vdi = measures.MultivariateVARDNNDirectionalInfluence()

            if opt.showmat:
                mat = vdi.plot(net=net, node_control=node_control, ex_control=ex_control, is_full_node=is_full_node)
            else:
                mat = vdi.calc(net=net, node_control=node_control, ex_control=ex_control, is_full_node=is_full_node)
            vdi_mat.append(mat)
        save_mat_file(opt, vdi_mat, CXnames, 'vddi')

    # calc and plot VARDNN Granger Causality
    vgc_mat = []
    if opt.vdgc:
        for i in range(len(CX)):
            cache_path = 'results' + os.sep + 'cache-vardnn-' + CXnames[i]
            net = models.MultivariateVARDNNetwork()
            if os.path.isdir(cache_path) and not opt.nocache:
                net.load(cache_path)
            else:
                ex_signal, node_control, ex_control, is_full_node = get_optional_signals(Cex, Cctrl, Cexctrl, i)
                net.init(x=CX[i], ex_signal=ex_signal, node_control=node_control, ex_control=ex_control,
                         lags=opt.lag, reg_l2=opt.l2)
                net.fit(x=CX[i], ex_signal=ex_signal, node_control=node_control, ex_control=ex_control,
                        epochs=opt.epoch, batch_size=int(CX[i].shape[1]/3))
                if not opt.nocache:
                    net.save(cache_path)

            vgc = measures.MultivariateVARDNNGrangerCausality()

            if opt.showmat:
                mat = vgc.plot(x=CX[i], ex_signal=ex_signal, node_control=node_control, ex_control=ex_control,
                               net=net, is_full_node=is_full_node)
            else:
                mat = vgc.calc(x=CX[i], ex_signal=ex_signal, node_control=node_control, ex_control=ex_control,
                               net=net, is_full_node=is_full_node)
            vgc_mat.append(mat)
        save_mat_file(opt, vgc_mat, CXnames, 'vdgc')

    plt.pause(1)
    input("Press Enter to exit...")
