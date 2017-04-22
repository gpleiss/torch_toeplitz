import torch
import torchtoeplitz as toeplitz


def test_toeplitz_constructs_tensor_from_vectors():
    c = torch.Tensor([1, 6, 4, 5])
    r = torch.Tensor([1, 2, 3, 7])

    res = toeplitz.toeplitz(c, r)
    actual = torch.Tensor([
        [1, 2, 3, 7],
        [6, 1, 2, 3],
        [4, 6, 1, 2],
        [5, 4, 6, 1],
    ])

    assert torch.equal(res, actual)


def test_sym_toeplitz_constructs_tensor_from_vector():
    c = torch.Tensor([1, 6, 4, 5])

    res = toeplitz.sym_toeplitz(c)
    actual = torch.Tensor([
        [1, 6, 4, 5],
        [6, 1, 6, 4],
        [4, 6, 1, 6],
        [5, 4, 6, 1],
    ])

    assert torch.equal(res, actual)
