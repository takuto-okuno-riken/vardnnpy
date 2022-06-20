import argparse


class ParseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = []

    def initialize(self):
        self.parser.add_argument('in_files', metavar='filename', nargs='+', help='filename of node status time-series (node x frames)')
        self.parser.add_argument('--vddi',  action='store_true', help='output VARDNN Directional Influence matrix result (<filename>_vddi.csv)')
        self.parser.add_argument('--vdgc', action='store_true', help='output VARDNN Granger Causality matrix result (<filename>_vdgc.csv)')
        self.parser.add_argument('--mvgc', action='store_true', help='output multivaliate Granger Causality matrix result (<filename>_mvgc.csv)')
        self.parser.add_argument('--pwgc', action='store_true', help='output pairwise Granger Causality matrix result (<filename>_pwgc.csv)')
        self.parser.add_argument('--fc', action='store_true', help='output Correlation matrix result (<filename>_fc.csv)')
        self.parser.add_argument('--pc', action='store_true', help='output Partial Correlation matrix result (<filename>_pc.csv)')
        self.parser.add_argument('--outpath', nargs=1, default='results', help='output files path (default:"results")')
        self.parser.add_argument('--format', type=int, default=0, help='save file format <type> 0:csv, 1:mat(each), 2:mat(all) (default:0)')
        self.parser.add_argument('--transform', type=int, default=0, help='input signal transform  0:raw, 1:sigmoid (default:0)')
        self.parser.add_argument('--transopt', type=float, default=float('NaN'), help='signal transform option (for type 1:centroid value)')
        self.parser.add_argument('--lag', type=int, default=3, help='time lag for mvGC, pwGC (default:3)')
        self.parser.add_argument('--ex', nargs=1, help='VARDNN exogenous input signal <files> (file1.csv[:file2.csv:...])')
        self.parser.add_argument('--nctrl', nargs=1, help='VARDNN node status control <files> (file1.csv[:file2.csv:...])')
        self.parser.add_argument('--ectrl', nargs=1, help='VARDNN exogenous input control <files> (file1.csv[:file2.csv:...])')
        self.parser.add_argument('--epoch', type=int, default=500, help='VARDNN training epoch number (default:500)')
        self.parser.add_argument('--l2', type=float, default=0.05, help='VARDNN training L2Regularization (default:0.05)')
        self.parser.add_argument('--showsig',  action='store_true', help='show node status signals of <filename>.csv')
        self.parser.add_argument('--showex',  action='store_true', help='show exogenous input signals of <file1>.csv')
        self.parser.add_argument('--showmat',  action='store_true', help='show result matrix of VARDNN-DI, VARDNN-GC, mvGC, pwGC, FC and PC')
        self.parser.add_argument('--nocache',  action='store_true', help='do not use cache file for VARDNN training')

    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt

