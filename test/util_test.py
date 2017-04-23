import torch
from torchtoeplitz import util

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
