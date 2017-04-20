from torch.utils.ffi import create_extension

ffi = create_extension(
    name='torchtoeplitz.libfft',
    headers=['torchtoeplitz/src/fft.h'],
    sources=['torchtoeplitz/src/fft.c'],
    verbose=True,
    with_cuda=False,
    package=True,
    relative_to=__file__,
)
