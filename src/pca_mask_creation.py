import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from osgeo import gdal

# Working directory
wd = "/home/mike/OneDrive/MSc Earth and Space Physics and Engineering/4th Semester/Thesis/Raw_Data/Lakes_Final_Data/L1/Resized"
lake = wd.split("/")
lake = lake[-2]

os.chdir(wd)
files = os.listdir(wd)

# Filter out non-image files (created from QGIS)
files = [
    file
    for file in files
    if not (
        os.path.isdir(file)
        or file.endswith(".aux")
        or file.endswith(".txt")
        or file.endswith(".xml")
    )
]
files.sort()


def read_bundle(filename):
    ds = gdal.Open(filename)
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray()
    ds = None
    return data


# Identify files
dataset = [os.path.join(wd, file) for file in files]
dataset.sort()

# Find specific bands or the DEM
for file in dataset:
    if "DEM" in file:
        dem = file
    elif "B02" in file:
        blue = file
    elif "B03" in file:
        green = file
    elif "B04" in file:
        red = file

# Read image data
blue = read_bundle(blue)
green = read_bundle(green)
red = read_bundle(red)

# Stack images to create an RGB image
images = np.stack((red, green, blue), axis=-1)
print(images.shape)  # Should output (1024, 1024, 3) if dimensions are correct

# Perform PCA on the flattened images
height, width, _ = images.shape
flat_images = images.reshape(height * width, 3)  # Flatten correctly

pca = PCA()
pca.fit(flat_images)

# Weighted sum of principal components based on explained variance
weighted_sum_image = np.zeros((height, width))

# Calculate weighted sum using all principal components
transformed_images = pca.transform(flat_images)
for i in range(transformed_images.shape[1]):
    component_image = transformed_images[:, i].reshape(height, width)
    weighted_sum_image += pca.explained_variance_ratio_[i] * component_image

# Normalize weighted_sum_image to be between 0 and 1
weighted_sum_image = (weighted_sum_image - np.min(weighted_sum_image)) / (
    np.max(weighted_sum_image) - np.min(weighted_sum_image)
)

# Threshold to create a mask for the lake
threshold = 0.3  # Adjust this threshold based on your data
lake_mask = weighted_sum_image < threshold

# Plot original image and lake mask
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(images)  # Plot the RGB image

plt.subplot(1, 2, 2)
plt.title("Lake Mask (Weighted Sum)")
plt.imshow(lake_mask, cmap="gray")

plt.show()
# breakpoint()
# Save the mask image
mask_name = lake + "_mask.png"
mask_dir = "/home/mike/OneDrive/MSc Earth and Space Physics and Engineering/4th Semester/Thesis/Raw_Data/Lakes_Final_Data/Masks"
if not os.path.exists(mask_dir):
    os.mkdir(mask_dir)
mask_path = os.path.join(mask_dir, mask_name)
plt.imsave(mask_path, lake_mask, cmap="gray")
print(f"Mask saved: {mask_path}")
