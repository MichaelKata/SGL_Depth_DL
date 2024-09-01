from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from network import UNET
from time import time
from natsort import natsorted

# Visualize Feature Maps
import torchvision.utils as vutils
from torchvision.utils import save_image


wd = "/home/mike/OneDrive/MSc Earth and Space Physics and Engineering/4th Semester/Thesis/Raw_Data/Lakes_Final_Data"
bundles_dir = os.path.join(wd, "Bundles")
vis_dir = os.path.join(wd, "Visualizer")
models_dir = os.path.join(wd, "Models")
figures_dir = os.path.join(wd, "Figures")
files = list()
for i in os.listdir(bundles_dir):
    files.append(os.path.join(bundles_dir, i))

files = natsorted(files)

# Split into training and validation sets
val_test_set = files[8:]
train_files = files[:8]

val_file = val_test_set[0]


print("Training files are:")
for i in train_files:
    print(i.split("/")[-1])

print(f"Validation file is: {val_file.split('/')[-1]}")


def read_bundle(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
        rd = data["Red"]
        gr = data["Green"]
        bl = data["Blue"]
        dm = data["DEM"]
        msk = data["Mask"]
    return rd, gr, bl, dm, msk


# Formulate a class to visualize the feature maps
class FeatureMapVisualizer:
    def __init__(self, model, layer_name, output_dir):
        self.model = model
        self.layer_name = layer_name
        self.features = None
        self.output_dir = output_dir
        self._register_hook()

    def _register_hook(self):
        def hook(module, input, output):
            self.features = output

        for name, module in self.model.named_modules():
            if name == self.layer_name:
                module.register_forward_hook(hook)

    def visualize(self, input_tensor, epoch, batch_idx):
        _ = self.model(input_tensor)
        if self.features is not None:
            feature_maps = self.features[0].cpu().detach()
            num_features = feature_maps.size(0)

            # Determine grid size
            grid_size = int(np.ceil(np.sqrt(num_features)))

            # Create a grid of subplots
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
            fig.suptitle(
                f"Feature Maps of {self.layer_name} (Epoch {epoch}, Batch {batch_idx})"
            )

            for i in range(num_features):
                ax = axes[i // grid_size, i % grid_size]
                ax.imshow(feature_maps[i].numpy(), cmap="viridis")
                ax.axis("off")

            # Remove empty subplots
            for i in range(num_features, grid_size * grid_size):
                fig.delaxes(axes[i // grid_size, i % grid_size])

            # Create the output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)

            # Save the figure
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    self.output_dir, f"feature_maps_epoch_{epoch}_batch_{batch_idx}.png"
                ),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()


###############################################################
######## TRAIN ################################################


# Dataset class
class LakeDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        self.data = []

        for file in self.file_list:
            red, green, blue, dem, mask = read_bundle(file)
            image = np.stack((red, green, blue), axis=2)
            self.data.append((image, mask, dem))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, mask, dem = self.data[idx]

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        dem = torch.tensor(dem, dtype=torch.float32).unsqueeze(0)

        return image, torch.cat((mask, dem), dim=0)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNET().to(device)
visualizer = visualizer = FeatureMapVisualizer(model, "downs.3.conv", vis_dir)
b_s = 4
criterion_mask = nn.BCEWithLogitsLoss()  # For binary classification (lake mask)
criterion_depth = nn.MSELoss()  # For regression (depth prediction)
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
num_epochs = 1000
validate_every = 10

# Different weightings for the mask and depth.
weight_mask_loss = 0.6  # Weight for mask loss
weight_depth_loss = 0.4  # Weight for depth loss

train_dataset = LakeDataset(train_files)
val_dataset = LakeDataset(
    [val_file]
)  # Because validation set needs to be a list for the DataLoader"

# Create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=b_s, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=b_s, shuffle=False)

# Lists to store losses
train_losses = []
train_depth_losses = []
train_extent_losses = []
val_losses = []
val_depth_losses = []
val_extent_losses = []

best_val_loss = float("inf")
best_model_state = None
save_model_after = int(num_epochs * 0.4)  # Save model after 40% of epochs

start = time()
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    epoch_depth_loss = 0
    epoch_extent_loss = 0

    for batch_idx, (data, targets) in enumerate(train_dataloader):
        data, targets = data.to(device), targets.to(device)
        targets_mask = targets[:, 0].unsqueeze(1)
        targets_depth = targets[:, 1].unsqueeze(1)

        outputs = model(data)
        outputs_mask_logits = outputs[:, 0].unsqueeze(1)
        outputs_depth = outputs[:, 1].unsqueeze(1)

        loss_mask = criterion_mask(outputs_mask_logits, targets_mask)
        loss_depth = criterion_depth(outputs_depth, targets_depth)

        weighted_loss_mask = weight_mask_loss * loss_mask
        weighted_loss_depth = weight_depth_loss * loss_depth
        loss = weighted_loss_mask + weighted_loss_depth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_depth_loss += loss_depth.item()
        epoch_extent_loss += loss_mask.item()

        if batch_idx % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}, Mask Loss: {loss_mask.item():.4f}, Depth Loss: {loss_depth.item():.4f}"
            )

        # Visualize feature maps for the first batch of the selected epoch
        if epoch == 5 and batch_idx == 0:
            visualizer.visualize(data[0].unsqueeze(0).to(device), epoch, batch_idx)

    average_loss = epoch_loss / len(train_dataloader)
    average_depth_loss = epoch_depth_loss / len(train_dataloader)
    average_extent_loss = epoch_extent_loss / len(train_dataloader)

    train_losses.append(average_loss)
    train_depth_losses.append(average_depth_loss)
    train_extent_losses.append(average_extent_loss)

    # Validation block
    if (epoch + 1) % validate_every == 0:
        model.eval()
        val_epoch_loss = 0
        val_epoch_depth_loss = 0
        val_epoch_extent_loss = 0

        with torch.no_grad():
            for val_batch_idx, (val_data, val_targets) in enumerate(val_dataloader):
                val_data, val_targets = val_data.to(device), val_targets.to(device)
                val_targets_mask = val_targets[:, 0].unsqueeze(1)
                val_targets_depth = val_targets[:, 1].unsqueeze(1)

                val_outputs = model(val_data)
                val_outputs_mask_logits = val_outputs[:, 0].unsqueeze(1)
                val_outputs_depth = val_outputs[:, 1].unsqueeze(1)

                val_loss_mask = criterion_mask(
                    val_outputs_mask_logits, val_targets_mask
                )
                val_loss_depth = criterion_depth(val_outputs_depth, val_targets_depth)

                weighted_val_loss_mask = weight_mask_loss * val_loss_mask
                weighted_val_loss_depth = weight_depth_loss * val_loss_depth
                val_loss = weighted_val_loss_mask + weighted_val_loss_depth

                val_epoch_loss += val_loss.item()
                val_epoch_depth_loss += val_loss_depth.item()
                val_epoch_extent_loss += val_loss_mask.item()

        average_val_loss = val_epoch_loss / len(val_dataloader)
        average_val_depth_loss = val_epoch_depth_loss / len(val_dataloader)
        average_val_extent_loss = val_epoch_extent_loss / len(val_dataloader)

        val_losses.append(average_val_loss)
        val_depth_losses.append(average_val_depth_loss)
        val_extent_losses.append(average_val_extent_loss)

        print(
            f"Validation - Epoch [{epoch+1}/{num_epochs}], Loss: {average_val_loss:.4f}, Depth Loss: {average_val_depth_loss:.4f}"
        )
        # Save the model on its best validation loss after 40% of the epochs
        if epoch + 1 >= save_model_after and average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            best_model_state = model.state_dict()

# Interpolate validation losses to match the length of training losses for plotting
val_losses_interpolated = np.interp(
    np.arange(num_epochs),
    np.arange(validate_every - 1, num_epochs, validate_every),
    val_losses,
)
val_depth_losses_interpolated = np.interp(
    np.arange(num_epochs),
    np.arange(validate_every - 1, num_epochs, validate_every),
    val_depth_losses,
)

val_extent_losses_interpolated = np.interp(
    np.arange(num_epochs),
    np.arange(validate_every - 1, num_epochs, validate_every),
    val_extent_losses,
)

if best_model_state:
    model_name = f"full_2_data_best_model_after_{save_model_after}_ep_{num_epochs}_bs_{b_s}_lr_{learning_rate}_w_depth_{weight_depth_loss}.pth"
    model_path = os.path.join(models_dir, model_name)
    torch.save(best_model_state, model_path)
    print(f"Best model saved at: {model_path}")

end = time()

duration = end - start
hours = duration // 3600
minutes = (duration % 3600) // 60
seconds = duration % 60

print(
    f"Training duration: {hours:.0f} hours, {minutes:.0f} minutes, {seconds:.0f} seconds"
)


figure_path = os.path.join(figures_dir, "Losses.png")
plt.plot(train_losses, label="Train Total Loss")
plt.plot(train_depth_losses, label="Train Depth Loss")
plt.plot(train_extent_losses, label="Train Extent Loss")
plt.plot(val_losses_interpolated, label="Validation Loss")
plt.plot(val_depth_losses_interpolated, label="Validation Depth Loss")
plt.plot(val_extent_losses_interpolated, label="Validation Extent Loss")
plt.legend()
plt.title("Training and Validation Losses")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(figure_path, dpi=600)
plt.show()

print(f"Depth Loss STD: {np.std(val_depth_losses):.3f}")
print(f"Extent Loss STD: {np.std(val_extent_losses):.3f}")
