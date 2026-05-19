import torch
import torch.nn.functional as F
from collections import OrderedDict
from dl_network import *    
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = model.to(device)

def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]

        if all(isinstance(k, str) and k.startswith("module.") for k in ckpt.keys()):
            new_sd = OrderedDict()
            for k, v in ckpt.items():
                new_sd[k.replace("module.", "", 1)] = v
            return new_sd

    return ckpt


def pad_to_multiple(x, multiple=64):
    # x: (B, C, H, W)
    B, C, H, W = x.shape
    H_new = ((H + multiple - 1) // multiple) * multiple
    W_new = ((W + multiple - 1) // multiple) * multiple
    pad_h = H_new - H
    pad_w = W_new - W
    x_pad = F.pad(x, (0, pad_w, 0, pad_h))
    return x_pad, H, W


# LOAD MODEL
def load_model(weights_path="trained_models/model_fk.pth"):
    """Load model weights called from MATLAB."""
    global model

    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = _extract_state_dict(ckpt)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print("[load_model] Missing keys:", missing)
    if unexpected:
        print("[load_model] Unexpected keys:", unexpected)

    model.eval()
    model.to(device)
    print("Model loaded from", weights_path)



# Run interface
@torch.no_grad()
def run_inference(real_mat, imag_mat, scale):

    real_mat = real_mat / scale
    imag_mat = imag_mat / scale

    # numpy -> torch
    real = torch.from_numpy(np.array(real_mat, dtype=np.float32))
    imag = torch.from_numpy(np.array(imag_mat, dtype=np.float32))

    # (1,2,H,W)
    inp = torch.stack([real, imag], dim=0).unsqueeze(0).to(device)

    ''' optional normalizaiton '''
    #mean = inp.mean()
    #std  = inp.std()

    #inp    = (inp - mean) / (std + 1e-8)


    out = model(inp)          # (1,2,H_pad,W_pad)

    out = out * scale

    out = out.squeeze(0)      # (2,H,W)
    pred_re = out[0].cpu().numpy()
    pred_im = out[1].cpu().numpy()

    return pred_re, pred_im
