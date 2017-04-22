import torch
from torch.autograd import Function
from .. import fft

class Mv(Function):
    def forward(self, c, r, v):
        assert c.ndimension() == 1
        assert r.ndimension() == 1
        assert v.ndimension() == 1
        assert len(c) == len(r)
        assert len(c) == len(v)
        assert c[0] == r[0]
        assert type(c) == type(r)
        assert type(c) == type(v)

        orig_size = len(c)
        r_reverse = r.index_select(0, torch.LongTensor(list(range(1, len(r))[::-1])))
        c.resize_(orig_size + len(r) - 1)
        c[orig_size:].copy_(r_reverse)

        v.resize_(2 * orig_size - 1)
        v[orig_size:].fill_(0)

        fft_c = fft.fft1(c)
        fft_v = fft.fft1(v)
        fft_product = torch.zeros(fft_c.size())

        fft_product[:, 0].addcmul_(fft_c[:, 0], fft_v[:, 0])
        fft_product[:, 0].addcmul_(-1, fft_c[:, 1], fft_v[:, 1])
        fft_product[:, 1].addcmul_(fft_c[:, 1], fft_v[:, 0])
        fft_product[:, 1].addcmul_(fft_c[:, 0], fft_v[:, 1])

        res = fft.ifft1(fft_product, c.size())
        c.resize_(orig_size)
        r.resize_(orig_size)
        v.resize_(orig_size)
        res.resize_(orig_size)
        return res

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
