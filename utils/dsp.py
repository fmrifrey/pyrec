import torch

def fftc(input_tensor, dims=None):
    
    # set default dimensions
    if dims is None:
        dims = list(range(input_tensor.dim()))
    elif any(dim >= input_tensor.dim() or dim < 0 for dim in dims):
        raise ValueError("dimensions out of range")
    
    # loop through dimensions and fft
    output = input_tensor.clone()
    for dim in dims:
        size = output.size(dim)
        # fftshift -> fft -> ifftshift in one go
        output = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(output, dim=dim), dim=dim), dim=dim) / torch.sqrt(torch.tensor(size, dtype=torch.float32))
    
    return output