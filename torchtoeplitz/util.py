import torch

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
