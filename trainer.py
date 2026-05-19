import scipy.io
import torch
import torch.optim as optim
from tqdm import tqdm
from utils.Loss import *
import dl_network
from Data_Loader import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device : {device}")
_lr = 2e-4
save_directory = "trained_models/model_fk.pth"

model = dl_network.model.to(device)
criterion = torch.nn.MSELoss().to(device)
#criterion = Correlation().to(device)
optimizer = optim.Adam(model.parameters(), lr=_lr)



# Load data

print("Loading data...")
data = scipy.io.loadmat("cached_data/training_cache_fk.mat")
inputs, outputs, B, H, W, U, scale = Load_Data(data, device)


# Training loop

num_epochs = 10
validation_freq = 5
print("Training start\n")
for epoch in range(num_epochs):
    with tqdm(total=B * U, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=100, dynamic_ncols=True) as pbar:

        for b in range(B):
            for u in range(U):

                inp    = inputs[b, :, :, u]      # (H, W) complex
                target = outputs[b, :, :, u]     # (H, W) complex

                inp    = inp / scale
                target = target / scale

                ''' optional normalization '''
                #mean = inp.mean()
                #std  = inp.std()

                #inp    = (inp - mean) / (std + 1e-8)
                #target = (target - mean) / (std + 1e-8)

                inp_tensor = complex_to_tensor(inp, device) # (1, 2, H, W)
                target_tensor = complex_to_tensor(target, device) # (1, 2, H, W)

                
                ''' optional normalization '''
                #patch_h, patch_w = 256, 256
                #H, W = inp_tensor.shape[-2:]

                #ys = torch.randint(0, H - patch_h + 1, (1,)).item()
                #xs = torch.randint(0, W - patch_w + 1, (1,)).item()

                #inp_tensor    = inp_tensor[..., ys:ys+patch_h, xs:xs+patch_w]
                #target_tensor = target_tensor[..., ys:ys+patch_h, xs:xs+patch_w]


                # Forward + backward
                optimizer.zero_grad()
                pred = model(inp_tensor)
                loss = criterion(pred, target_tensor)
                loss.backward()
                optimizer.step()

                # update tqdm
                pbar.update(1)
                pbar.set_postfix({"loss": float(loss.item())})
                
        if (epoch + 1) % validation_freq == 0:
            torch.save(model.state_dict(), save_directory)
            print(f"Saved model to {save_directory}")
            torch.cuda.empty_cache()

print("\nTraining complete.")
model.eval()

torch.save(model.state_dict(), save_directory)
print(f"Saved model to {save_directory}")