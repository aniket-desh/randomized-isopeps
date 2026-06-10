import numpy as np

class StoppingRule(object):

    def __init__(self, outputdim, mindim, maxdim, cutoff):
        self.outputdim = outputdim
        self.mindim = mindim
        self.maxdim = maxdim
        self.cutoff = cutoff

    def is_truncation(self):
        return not ((self.outputdim is None) and (self.cutoff is None))

def Cutoff(cutoff, mindim = 1, maxdim = np.inf):
    return StoppingRule(None, mindim, maxdim, cutoff)

def FixedDimension(dim):
    return StoppingRule(dim, 1, np.inf, None)

def no_truncation():
    return StoppingRule(None, None, np.inf, None)
