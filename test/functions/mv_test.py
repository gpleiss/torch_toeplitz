import math
import torch
import torchtoeplitz as toeplitz
from torch.autograd import Variable


def approx_equal(self, other, epsilon=1e-5):
    return torch.max((self.data - other.data).abs()) <= epsilon


def test_mv_performs_toeplitz_matrix_vector_multiplication():
    c = Variable(torch.randn(5))
    r = Variable(torch.randn(5))
    r.data[0:1].fill_(c.data[0])
    v = Variable(torch.randn(5))

    m = Variable(toeplitz.toeplitz(c.data, r.data))
    actual = torch.mv(m, v)

    res = toeplitz.functions.mv(c, r, v)
    assert approx_equal(actual, res)


def test_sym_mv_performs_toeplitz_matrix_vector_multiplication():
    c = Variable(torch.randn(5))
    v = Variable(torch.randn(5))

    m = Variable(toeplitz.sym_toeplitz(c.data))
    actual = torch.mv(m, v)

    res = toeplitz.functions.sym_mv(c, v)
    assert approx_equal(actual, res)
