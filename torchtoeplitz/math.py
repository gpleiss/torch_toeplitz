import torch
from . import util
from . import fft

def mv(c, r, v):
    assert c.ndimension() == 1
    assert r.ndimension() == 1
    assert v.ndimension() == 1
    assert len(c) == len(r)
    assert len(c) == len(v)
    assert c[0] == r[0]
    assert type(c) == type(r)
    assert type(c) == type(v)

    orig_size = len(c)
    r_reverse = util.reverse(r[1:])
    c.resize_(orig_size + len(r_reverse))
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


def sym_minvv(c, v, eps=1e-20):
    assert c.ndimension() == 1
    assert v.ndimension() == 1
    assert len(c) == len(v)
    assert type(c) == type(v)

    # Initialize variables of conjugate gradients algorithm
    x = v.clone().fill_(0) # x_0 = 0
    r = v.clone() #r_0 = v - mat * x_0
    p = r.clone() # p_0 = r_0

    # Store products
    curr_r_inner = torch.dot(r, r) # r_i^T r_i
    next_r_inner = None

    for i in range(100):
        # Store products
        mat_p = mv(c, c, p) # mat * p_i

        alpha = curr_r_inner / (torch.dot(p, mat_p)) # alpha_i = r_i^T r_i / (p_i^T mat p_i)
        x.add_(alpha, p) # x_i+1 = x_i + alpha_i * p
        r.add_(-alpha, mat_p) # r_i - alpha_i mat p_i

        # Exit condition
        if r.norm() / len(r) <= eps:
            break

        next_r_inner = torch.dot(r, r)
        beta = next_r_inner / (curr_r_inner) # beta_i = (r_i+1^T r_i+1) / (r_i^T r_i)
        p.mul_(beta)
        p.add_(r) # p_i+1 = r_i+1 + beta_i p_i

        curr_r_inner = next_r_inner
        next_r_inner = None

    return x
