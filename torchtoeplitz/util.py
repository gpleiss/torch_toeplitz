import torch
from torch.autograd import Variable

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


'''
    Reverses a tensor
    Args:
        - input: tensor to reverse
        - dim: dimension to reverse on
    Returns:
        - reversed input
'''
def reverse(input, dim=0):
    reverse_index = torch.LongTensor(list(range(input.size(dim))[::-1]))
    return input.index_select(dim, reverse_index)


'''
    Computes a reverse cumulative sum
    Args:
        - input: tensor
        - dim: dimension to reverse on
    Returns:
        - rcumsum on input
'''
def rcumsum(input, dim=0):
    reverse_index = torch.LongTensor(list(range(input.size(dim))[::-1]))
    return torch.index_select(input, dim, reverse_index).cumsum(dim).index_select(dim, reverse_index)


'''
    Determines if two tensors are approximately equal
    Args:
        - self: tensor
        - other: tensor
    Returns:
        - bool
'''
def approx_equal(self, other, epsilon=1e-5):
    if isinstance(self, Variable):
        self = self.data
    if isinstance(other, Variable):
        other = other.data
    return torch.max((self - other).abs()) <= epsilon
