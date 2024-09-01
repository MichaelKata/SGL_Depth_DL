import pickle
import numpy as np
from osgeo import gdal
import os
import re
from natsort import natsorted
import matplotlib.pyplot as plt
from PIL import Image

gdal.DontUseExceptions()

wd = "/home/mike/OneDrive/MSc Earth and Space Physics and Engineering/4th Semester/Thesis/Raw_Data/Lakes_Final_Data"
bundles_dir = os.path.join(wd, "Bundles")
if os.path.exists(bundles_dir) == False:
    os.mkdir(bundles_dir)

pattern = re.compile(r"^L")
lake_dirs = list()


def open_tiff(file):
    ds = gdal.Open(file)
    band = ds.GetRasterBand(1)
    band = band.ReadAsArray()
    ds = None
    return band


def open_mask(file):
    file = Image.open(file)
    file = np.array(file)
    file = file[:, :, 0] / 255
    return file


def create_bundle(red_tmp, green_tmp, blue_tmp, dem_tmp, mask_tmp, path_tmp):
    data = dict()
    data["Red"] = red_tmp
    data["Green"] = green_tmp
    data["Blue"] = blue_tmp
    data["DEM"] = dem_tmp
    data["Mask"] = mask_tmp
    with open(path_tmp, "wb") as f:
        pickle.dump(data, f)
    return data


surface_reference = {
    "L1": 1168.17,
    "L2": 1242.35,
    "L3": 1293.11,
    "L4": 1210.42,
    "L5": 1319.59,
    "L6": 1123.94,
    "L7": 1462.63,
    "L8": 1202.03,
    "L9": 1464.76,
    "L10": 1385.41,
    "L11": 1312.63,
    "L12": 1300.62,
    "L13": 900.61,
}

###########################################################################################

for i in os.listdir(wd):
    if pattern.match(i):
        lake_dirs.append(i)
lake_dirs.sort()
element = lake_dirs.pop(1)
lake_dirs.append(element)
lake_dirs = natsorted(lake_dirs)
print(lake_dirs)


lake_files = dict()
for i in lake_dirs:
    resized_dir = "Resized"
    path = os.path.join(wd, i, resized_dir)
    temp_list = list()
    for i2 in os.listdir(path):
        temp_list.append(os.path.join(path, i2))
    lake_files[i] = temp_list


for i in lake_files.keys():
    temp_list = lake_files[i]
    temp_list.sort()
    red = open_tiff(temp_list[2])
    green = open_tiff(temp_list[1])
    blue = open_tiff(temp_list[0])
    dem = open_tiff(temp_list[3])
    dem = dem - surface_reference[i]
    # image = np.stack((red, green, blue), axis=-1)
    mask = open_mask(temp_list[4])
    name = i + ".pkl"
    path = os.path.join(bundles_dir, name)
    create_bundle(red, green, blue, dem, mask, path)

print(f"Created the lake bundles in:\n{bundles_dir}")
