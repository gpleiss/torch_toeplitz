import torch
from torch.autograd import Function
from .. import util
from .. import math

class SymMinvv(Function):
    def forward(self, c, v):
        self.c = c
        self.v = v
        return math.sym_minvv(self.c, self.v)


'''
    Performs toeplitz matrix inverse-vector multiplication
    Precondition: toeplitz matrix must be positive definite
    Args:
        - c (vector n) - column of toeplitz matrix
        - v (vector n) - vector for multiplication
    Returns:
        - Vector (n)
'''
def sym_minvv(c, v):
    return SymMinvv()(c, v)
