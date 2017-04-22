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

        r_reverse = r.index_select(0, torch.LongTensor(list(range(1, len(r))[::-1])))
        circ_vector = torch.cat((c, r_reverse), 0)
        v_aug = torch.cat((v, torch.zeros(len(r) - 1)), 0)

        fft_circ_vector = fft.fft1(circ_vector)
        fft_v_aug = fft.fft1(v_aug)
        fft_product = torch.zeros(fft_circ_vector.size())

        fft_product[:, 0].addcmul_(fft_circ_vector[:, 0], fft_v_aug[:, 0])
        fft_product[:, 0].addcmul_(-1, fft_circ_vector[:, 1], fft_v_aug[:, 1])
        fft_product[:, 1].addcmul_(fft_circ_vector[:, 1], fft_v_aug[:, 0])
        fft_product[:, 1].addcmul_(fft_circ_vector[:, 0], fft_v_aug[:, 1])

        res = fft.ifft1(fft_product, circ_vector.size())
        res.resize_(len(c))
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
