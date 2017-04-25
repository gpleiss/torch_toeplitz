import torch
from torch.autograd import Function
from .. import util
from .. import math

class Mv(Function):
    def forward(self, c, r, v):
        self.c = c
        self.r = r
        self.v = v
        return math.mv(self.c, self.r, self.v)

    def backward(self, grad_output):
        di_dc = util.rcumsum(self.v * grad_output)
        di_dr = di_dc.clone()
        di_dv = math.mv(self.r, self.c, grad_output)

        return di_dc, di_dr, di_dv

'''
    Performs toeplitz matrix-vector multiplication
    Args:
        - c (vector n) - column of toeplitz matrix
        - r (vector n-1) - row of toeplitz matrix
        - v (vector n) - vector for multiplication
    Returns:
        - Vector (n)
'''
def mv(c, r, v):
    return Mv()(c, r, v)

'''
    Performs toeplitz matrix-vector multiplication for symmetric toeplitz matrix
    Args:
        - c (vector n) - column of toeplitz matrix
        - v (vector n) - vector for multiplication
    Returns:
        - Vector (n)
'''
def sym_mv(c, v):
    return Mv()(c, c, v)
