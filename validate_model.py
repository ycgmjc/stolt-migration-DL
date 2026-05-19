import numpy as np
import scipy.io
import torch
import matplotlib.pyplot as plt

from dl_fk_multi import *
import load_fk_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device : {device}")


def load_beams_cached(path="cached_data/beams_cached.mat"):
    mat = scipy.io.loadmat(path)
    
    full_sch_data = mat["full_sch_data"]
    start_times   = float(mat["start_times"].squeeze())
    tshift        = float(mat["tshift"].squeeze())
    fs            = float(mat["fs"].squeeze())
    f0            = float(mat["f0"].squeeze())
    rxAptPos      = mat["rxAptPos"]
    c             = float(mat["c"].squeeze())

    print("[beams_cached] full_sch_data shape:", full_sch_data.shape)
    print("[beams_cached] rxAptPos shape:", rxAptPos.shape)

    return full_sch_data, start_times, tshift, fs, f0, rxAptPos, c

def load_scale(path="cached_data/training_cache_fk.mat"):
    mat = scipy.io.loadmat(path)
    return float(mat.get("scale", 1.0))


def validate():
    # Load RAW RF + params from beams_cached.mat (MATLAB cache)
    (full_sch_data, start_times, tshift, fs, f0, rxAptPos, c) = load_beams_cached()

    print("Loaded full_sch_data:", full_sch_data.shape)

    # Load DL scale once and set it in python model wrapper
    scale = load_scale()

    # Run DL-FK pipeline
    start_time_stolt = start_times - tshift
    f_params = {"fs": fs, "f0": f0}

    foc_data, t_axis, x_axis = dl_fk_multi(
        full_sch_data,
        f_params,
        rxAptPos,
        scale,
        start_time=start_time_stolt,
        speed_of_sound=c,
        )


    # Envelope + normalize
    env = np.abs(foc_data)
    env = env / (env.max() + 1e-12)

    ax_data = t_axis * c / 2.0

    # Lateral upsampling
    ncols = env.shape[1]
    original_spacing  = np.linspace(0, 1, ncols)
    upsampled_spacing = np.linspace(0, 1, ncols * 4)

    env_interp = np.vstack([
        np.interp(upsampled_spacing, original_spacing, env[r])
        for r in range(env.shape[0])
    ])

    x_us = np.interp(upsampled_spacing, original_spacing, x_axis)

    # Visualization
    plt.figure(figsize=(6, 8))
    plt.imshow(
        20 * np.log10(env_interp + 1e-12),
        extent=[x_us.min(), x_us.max(), ax_data.max(), ax_data.min()],
        cmap="gray",
        aspect="auto",
    )
    plt.clim([-80, 0])
    plt.axis("off")
    plt.title("Validation: DL-FK using beams_cached.mat")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    validate()