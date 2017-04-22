import torch
from . import fft

'''
    Constructs tensor version of toeplitz matrix from column vector
    Args:
        - c (vector n) - column of toeplitz matrix
        - r (vector n-1) - row of toeplitz matrix
    Returns:
        - Matrix (n x n) - matrix representation
'''
def toeplitz(c, r):
    assert c.ndimension() == 1
    assert r.ndimension() == 1
    assert c[0] == r[0]
    assert len(c) == len(r)
    assert type(c) == type(r)

    res = torch.Tensor(len(c), len(c)).type_as(c)
    for i, val in enumerate(c):
        for j in range(len(c) - i):
            res[j+i, j] = val
    for i, val in list(enumerate(r))[1:]:
        for j in range(len(r) - i):
            res[j, j+i] = val
    return res


'''
    Constructs tensor version of symmetric toeplitz matrix from column vector
    Args:
        - c (vector n) - column of toeplitz matrix
    Returns:
        - Matrix (n x n) - matrix representation
'''
def sym_toeplitz(c):
    return toeplitz(c, c)



