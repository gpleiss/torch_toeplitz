import math
import torch
import torchtoeplitz as toeplitz
from torchtoeplitz import util
from torch.autograd import Variable


def test_sym_minvv_performs_symmetric_toeplitz_matrix_inverse_vector_multiplication():
    c = Variable(torch.Tensor([4, -2, 0, -1, 1])) # Some random vector I found
            # that produces a pos-def toeplitz matrix
    v = Variable(torch.randn(5))

    m_inv = Variable(util.sym_toeplitz(c.data).inverse())
    actual = torch.mv(m_inv, v)

    res = toeplitz.functions.sym_minvv(c, v)
    print(actual)
    print(res)
    assert util.approx_equal(actual, res)
    assert False
