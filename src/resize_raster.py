from osgeo import gdal
import os

gdal.DontUseExceptions()

wd = "/home/mike/OneDrive/MSc Earth and Space Physics and Engineering/4th Semester/Thesis/Raw_Data/Lakes_Final_Data/L13"

os.chdir(wd)
files = os.listdir(wd)

resized_folder = os.path.join(wd, "Resized")
if not os.path.exists(resized_folder):
    os.mkdir(resized_folder)

# Filter out files containing "aux" in their names
for file in files:
    if file.endswith(".aux") or file.endswith(".txt") or file.endswith(".xml"):
        files.remove(file)

files.sort()


def resize_raster(input_dem, output_dem, width, height):
    print(f"Resizing to: {width}x{height}")
    dataset = gdal.Open(input_dem)
    gdal.Warp(
        output_dem,
        dataset,
        width=width,
        height=height,
        resampleAlg=gdal.GRA_Bilinear,
    )


width = 1024
height = 1024

# Resize all the images in the resized_folder
for file in files:
    print(os.path.abspath(file))
    out_file = os.path.join(resized_folder, file)
    print(out_file)
    resize_raster(file, out_file, width, height)

print("Finished resizing all the files!")
