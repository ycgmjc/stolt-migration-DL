import torch
import dl_network
import numpy as np

model = dl_network.model

def complex_to_tensor(slice_c, device):
    '''
    (H, W) -> (1, 2, H, W)
    '''
    if isinstance(slice_c, np.ndarray):
        real = torch.from_numpy(slice_c.real.astype(np.float32))
        imag = torch.from_numpy(slice_c.imag.astype(np.float32))
    else:  
        real = slice_c.real.float()
        imag = slice_c.imag.float()

    x = torch.stack([real, imag], dim=0)  # (2, H, W)
    x = x.unsqueeze(0).to(device, non_blocking=True)         # (1, 2, H, W)
    return x

def Load_Data(data, device):
    inputs_c  = data["inputs"]
    outputs_c = data["outputs"]

    print("Loaded shapes directly from scipy:")
    print("  inputs_c.shape :", inputs_c.shape)
    print("  outputs_c.shape:", outputs_c.shape)

    inputs_t  = torch.from_numpy(inputs_c)
    outputs_t = torch.from_numpy(outputs_c)

    B, H, W, U = inputs_c.shape
    
    scale_raw = data.get("scale", 1.0)
    
    # If numpy array extract the single item
    if isinstance(scale_raw, np.ndarray):
        scale = float(scale_raw.item())
    else:
        scale = float(scale_raw)
        
    print("Loaded scale =", scale)

    return inputs_t, outputs_t, B, H, W, U, scale