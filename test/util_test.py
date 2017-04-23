import torch
from torchtoeplitz import util


def test_toeplitz_constructs_tensor_from_vectors():
    c = torch.Tensor([1, 6, 4, 5])
    r = torch.Tensor([1, 2, 3, 7])

    res = util.toeplitz(c, r)
    actual = torch.Tensor([
        [1, 2, 3, 7],
        [6, 1, 2, 3],
        [4, 6, 1, 2],
        [5, 4, 6, 1],
    ])

    assert torch.equal(res, actual)


def test_sym_toeplitz_constructs_tensor_from_vector():
    c = torch.Tensor([1, 6, 4, 5])

    res = util.sym_toeplitz(c)
    actual = torch.Tensor([
        [1, 6, 4, 5],
        [6, 1, 6, 4],
        [4, 6, 1, 6],
        [5, 4, 6, 1],
    ])

    assert torch.equal(res, actual)

def test_reverse():
    input = torch.Tensor([
        [1, 2, 3],
        [4, 5, 6],
    ])
    res = torch.Tensor([
        [3, 2, 1],
        [6, 5, 4],
    ])
    assert torch.equal(util.reverse(input, dim=1), res)


def test_rcumsum():
    input = torch.Tensor([
        [1, 2, 3],
        [4, 5, 6],
    ])
    res = torch.Tensor([
        [6, 5, 3],
        [15, 11, 6],
    ])
    assert torch.equal(util.rcumsum(input, dim=1), res)
