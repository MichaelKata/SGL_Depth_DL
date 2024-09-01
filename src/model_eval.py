import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import torch
from prettytable import PrettyTable
from network import UNET
import rasterio
from PIL import Image
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    jaccard_score,
    confusion_matrix,
)
import pandas as pd
from natsort import natsorted
from scipy.ndimage import binary_dilation

# Define paths
wd = "/home/mike/OneDrive/MSc Earth and Space Physics and Engineering/4th Semester/Thesis/Raw_Data/Lakes_Final_Data"
bundles_dir = os.path.join(wd, "Bundles")
models_dir = os.path.join(wd, "Models")
best_model = "Best/best_ep_1000_bs_3_lr_0.001.pth"
best_new = "Best/best_model_after_900_ep_1000_bs_4_lr_0.001_w_depth_0.4_l_1_7.pth"
# best_new_full = "full_2_data_best_model_after_400_ep_1000_bs_4_lr_0.001_w_depth_0.4.pth"
model_name = best_new
model_path = os.path.join(models_dir, model_name)

# Get the test file
files = [os.path.join(bundles_dir, f) for f in os.listdir(bundles_dir)]
files = natsorted(files)

for idx, file in enumerate(files):
    print(f"{idx}: {file}")

print(files)

test_file = files[9]


lake_id = test_file.split("/")[-1]
lake_id = lake_id.split(".")[0]

print(f"Test file: {test_file}")


def read_bundle(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
        rd = data["Red"]
        gr = data["Green"]
        bl = data["Blue"]
        dm = data["DEM"]
        msk = data["Mask"]
    return rd, gr, bl, dm, msk


def open_raster(filename):
    with rasterio.open(filename) as src:
        band = src.read(1)
        band = np.array(band)
        return band


# Set up device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
model = UNET().to(device)
model.load_state_dict(torch.load(model_path))

model.eval()

# Prepare the test data
red, green, blue, dem, mask = read_bundle(test_file)

################################################################################################
# RANDOM NOISE FOR A MASK TO CHECK THE CORRESPONDING DEPTH OUTPUT ##############################
# shape = (1024, 1024)
# random_array = np.random.rand(*shape)
# mask = random_array
################################################################################################


image = np.stack((red, green, blue), axis=2)

data = (
    torch.tensor(image, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(device)
)
targets_mask = (
    torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
)
targets_depth = (
    torch.tensor(dem, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
)


# Get model predictions
with torch.no_grad():
    outputs = model(data)
    outputs_mask_logits = outputs[:, 0].unsqueeze(1)  # Logits for mask prediction
    outputs_depth = outputs[:, 1].unsqueeze(1)  # Depth prediction

    outputs_mask_binary = (
        (outputs_mask_logits >= 0).float().cpu().numpy()[0, 0]
    )  # Binary mask prediction
    outputs_depth = outputs_depth.cpu().numpy()[0, 0]  # Depth prediction

    targets_mask = targets_mask.cpu().numpy()[0, 0]  # Ground truth mask
    targets_depth = targets_depth.cpu().numpy()[0, 0]  # Ground truth depth

# Calculate differences
mask_diff = np.abs(targets_mask - outputs_mask_binary)
depth_diff = np.abs(targets_depth - outputs_depth)

# Find minimum and maximum depth values
min_depth = np.min([targets_depth.min(), outputs_depth.min()])
max_depth = np.max([targets_depth.max(), outputs_depth.max()])
min_diff = np.min(depth_diff)
max_diff = np.max(depth_diff)

################################################################################
################### PLOTTING ###################################################
sns.set_style("whitegrid")
temp_plot_dir = "/home/mike/OneDrive/MSc Earth and Space Physics and Engineering/4th Semester/Thesis/Thesis Document/Pictures/Results/Final_Model"
extent_plot_path = os.path.join(temp_plot_dir, lake_id + "_extent.png")
depth_plot_path = os.path.join(temp_plot_dir, lake_id + "_depth.png")


def plot_image_mask(ax, data, title, cmap, vmin=None, vmax=None):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    return im


# Mask plots
fig_mask, axes_mask = plt.subplots(1, 3, figsize=(20, 7))

im_mask_ground = plot_image_mask(axes_mask[0], targets_mask, "Label Mask", "gray")
plt.colorbar(im_mask_ground, ax=axes_mask[0], fraction=0.046, pad=0.04)
im_mask_pred = plot_image_mask(
    axes_mask[1], outputs_mask_binary, "Predicted Mask (Binary)", "gray"
)
plt.colorbar(im_mask_pred, ax=axes_mask[1], fraction=0.046, pad=0.04)
im_mask_diff = plot_image_mask(axes_mask[2], mask_diff, "Mask Difference", "gray")
plt.colorbar(im_mask_diff, ax=axes_mask[2], fraction=0.046, pad=0.04)
plt.tight_layout(pad=1)
fig_mask.savefig(extent_plot_path)
# plt.show()
plt.close()


# Plot depths
def plot_image(ax, data, title, cmap, vmin=None, vmax=None):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    return im


fig_depth, axes_depth = plt.subplots(1, 3, figsize=(20, 7))

im_target = plot_image(
    axes_depth[0], targets_depth, "Label Depth (m)", "terrain", vmin=min_depth, vmax=5
)
im_output = plot_image(
    axes_depth[1],
    outputs_depth,
    "Predicted Depth (m)",
    "terrain",
    vmin=min_depth,
    vmax=5,
)
im_diff = plot_image(
    axes_depth[2], depth_diff, "Depth Difference (m)", "turbo", vmin=0, vmax=5
)

plt.colorbar(im_target, ax=axes_depth[0], fraction=0.046, pad=0.04)
plt.colorbar(im_output, ax=axes_depth[1], fraction=0.046, pad=0.04)
plt.colorbar(im_diff, ax=axes_depth[2], fraction=0.046, pad=0.04)
plt.tight_layout(pad=1)
fig_depth.savefig(depth_plot_path)
# plt.show()
plt.close()
#############################################################################
# Calculate statistics for the lake
sep = 100 * "#"
print(sep)
# Calculate Mean Absolute Error for the lake area only
lake_area_indices = targets_mask == 1  # Get indices of the lake area
mae_depth_lake = np.mean(depth_diff[lake_area_indices])
mae_depth = np.mean(depth_diff)

print(sep)


# Function to see the model parameters clearly
def count_parameters(model, wd):
    headers = ["Modules", "Parameters"]
    table = PrettyTable(headers)
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)

    # Convert PrettyTable to Pandas DataFrame
    df = pd.DataFrame(data=table.rows, columns=table.field_names)

    # Save DataFrame to Excel file
    table_path = os.path.join(wd, "Model_Parameters.xlsx")
    df.to_excel(table_path, index=False)

    print(f"Total Trainable Parameters: {total_params}")
    return total_params


count_parameters(model, wd)

print(sep)


# Function to calculate mask metrics
def calculate_mask_metrics(target_mask, predicted_mask):
    # Flatten the masks
    target_mask_flat = target_mask.flatten()
    predicted_mask_flat = predicted_mask.flatten()

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(target_mask_flat, predicted_mask_flat).ravel()

    # Calculate metrics
    accuracy = accuracy_score(target_mask_flat, predicted_mask_flat)
    precision = precision_score(target_mask_flat, predicted_mask_flat)
    recall = recall_score(target_mask_flat, predicted_mask_flat)
    f1 = f1_score(target_mask_flat, predicted_mask_flat)
    iou = jaccard_score(target_mask_flat, predicted_mask_flat)

    # Calculate specificity (true negative rate)
    specificity = tn / (tn + fp)

    return {
        "True Positives": tp,
        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall (Sensitivity)": recall,
        "Specificity": specificity,
        "F1 Score": f1,
        "IoU (Jaccard)": iou,
    }


# Depth Statistics
def calculate_statistics(target_depth, predicted_depth, target_mask, predicted_mask):
    # Calculate depth statistics
    lake_area_indices = target_mask == 1
    depth_diff = np.abs(target_depth - predicted_depth)
    lake_depth_diff = depth_diff[lake_area_indices]

    mae = np.mean(lake_depth_diff)
    std_error = np.std(lake_depth_diff)
    mse = np.mean(lake_depth_diff**2)
    rmse = np.sqrt(mse)

    mask_positive = np.logical_and(target_mask == 1, target_depth != 0)
    mape_values = np.abs(target_depth - predicted_depth) / target_depth
    mape_values = mape_values[mask_positive]
    mape = np.mean(mape_values) * 100

    max_error = np.max(lake_depth_diff)

    # Calculate mask metrics
    mask_metrics = calculate_mask_metrics(target_mask, predicted_mask)

    return {
        "Depth Metrics": {
            "MAE": mae,
            "STD of Error": std_error,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape,
            "Max Error": max_error,
        },
        "Mask Metrics": mask_metrics,
    }


stats = calculate_statistics(
    targets_depth, outputs_depth, targets_mask, outputs_mask_binary
)

print("Depth Metrics:")
for stat_name, value in stats["Depth Metrics"].items():
    print(f"{stat_name}: {value:.3f}")

print(sep)

print("\nMask Metrics:")
for stat_name, value in stats["Mask Metrics"].items():
    print(f"{stat_name}: {value:.3f}")

print("Evaluation complete.")


#######################################################################################################################
# Testing RTE
band = green
band = np.max(band) - band
A_d_red = 0.1347  # Lake bottom albedo/reflectance
R_infinity_red = 0.0254  # Reflectance of optically deep water
K_d_red = 0.4075875  # Coefficient for spectral radiance loss
g_red = 2.75 * K_d_red
A_d_green = 0.2055
R_infinity_green = 0.0474
K_d_green = 0.07636
g_green = 2.75 * K_d_green


# Calculating the depth for each pixel
def rte(Rw, R_infinity, A_d):
    return np.log(R_infinity / Rw) - np.log(A_d / R_infinity)


# Compute RTE depth
rte_depth = rte(band, R_infinity_green, A_d_green)
rte_depth_row = rte_depth
rte_depth = np.where(mask == 1, rte_depth, 0)
rte_depth = np.ma.masked_where(rte_depth == 0, rte_depth)
unet_depth = np.where(mask == 1, outputs_depth, 0)
unet_depth = np.ma.masked_where(unet_depth == 0, unet_depth)
v_min = -6
v_max = 6

# Create a figure with 2 subplots side by side
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

cax1 = ax[0].imshow(unet_depth, cmap="bwr", vmin=v_min, vmax=v_max)
ax[0].set_title("U-Net")
fig.colorbar(cax1, ax=ax[0])

cax2 = ax[1].imshow(rte_depth, cmap="bwr", vmin=v_min, vmax=v_max)
ax[1].set_title("RTE")
fig.colorbar(cax2, ax=ax[1])

# plt.show()
plt.close()
print(np.mean(depth_diff))

middle_column_index = outputs_depth.shape[1] // 2
dem_row = outputs_depth[:, middle_column_index]
rte_row = rte_depth_row[:, middle_column_index]

subsurface = dem_row <= 0

# Keep only the lake shore and less values (DEM <= 0)
dem_row = dem_row[subsurface]
rte_row = rte_row[subsurface]

plt.plot(dem_row, label="DEM")
plt.plot(rte_row, label="RTE")
plt.legend()
plt.show()
