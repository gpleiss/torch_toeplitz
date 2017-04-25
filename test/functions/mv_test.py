import math
import torch
import torchtoeplitz as toeplitz
from torchtoeplitz import util
from torch.autograd import Variable


def test_mv_performs_toeplitz_matrix_vector_multiplication():
    c = Variable(torch.randn(5))
    r = Variable(torch.randn(5))
    r.data[0:1].fill_(c.data[0])
    v = Variable(torch.randn(5))

    m = Variable(util.toeplitz(c.data, r.data))
    actual = torch.mv(m, v)

    res = toeplitz.functions.mv(c, r, v)
    assert util.approx_equal(actual, res)


def test_sym_mv_performs_toeplitz_matrix_vector_multiplication():
    c = Variable(torch.randn(5))
    v = Variable(torch.randn(5))

    m = Variable(util.sym_toeplitz(c.data))
    actual = torch.mv(m, v)

    res = toeplitz.functions.sym_mv(c, v)
    assert util.approx_equal(actual, res)

def test_mv_backwards_performs_toeplitz_matrix_vector_multiplication():
    c = Variable(torch.randn(5), requires_grad=True)
    r = Variable(torch.randn(5), requires_grad=True)
    r.data[0:1].fill_(c.data[0])
    v = Variable(torch.randn(5), requires_grad=True)

    m = Variable(util.toeplitz(c.data, r.data), requires_grad=True)
    actual = torch.mv(m, v).sum()
    actual.backward()

    actual_v_grad = v.grad.data.clone()
    actual_cr_grad = util.rcumsum(m.grad.data[0])

    v.grad.data.fill_(0)

    res = toeplitz.functions.mv(c, r, v).sum()
    res.backward()
    assert util.approx_equal(v.grad.data, actual_v_grad)
    assert util.approx_equal(r.grad.data, actual_cr_grad)
    assert util.approx_equal(c.grad.data, actual_cr_grad)
