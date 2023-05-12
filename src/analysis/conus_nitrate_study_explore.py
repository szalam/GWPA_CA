#%%
import sys
sys.path.insert(0,'src')
import config
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
# %%

df = pd.read_csv(config.data_raw / 'data_from_secondary_source_papers/ransom_conus_2022/National_NO3/Inputs/training_and_holdout_data.txt',delimiter='\t')
# %%

# Create a Point object for each row in the DataFrame
geometry = [Point(xy) for xy in zip(df.DEC_LONG_VA, df.DEC_LAT_VA)]

# Create a GeoDataFrame from the DataFrame and the Point objects
gdf = gpd.GeoDataFrame(df, geometry=geometry)

# Set the coordinate reference system (CRS) to WGS84
gdf.crs = {'init': 'epsg:4326'}

# Print the first few rows of the GeoDataFrame
print(gdf.head())

# %%
# Read the shapefile
shapefile = gpd.read_file(config.data_raw / 'shapefile/cv.shp')

# Clip the GeoDataFrame to the shape of the shapefile
clipped_gdf = gpd.clip(gdf, shapefile)

# Make sure the GeoDataFrame and the shapefile are in the same CRS
clipped_gdf = clipped_gdf.to_crs(shapefile.crs)

# Print the first few rows of the clipped GeoDataFrame
print(clipped_gdf.head())

# %%
clipped_gdf.plot()
# %%
clipped_gdf.shape
# %%

import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt

# Replace the file path with the path to your TIF file
file_path = '/Users/szalam/Downloads/Sarfaraz_test_psscene_analytic_sr_udm2/files/20230110_042730_39_2413_3B_AnalyticMS_SR_clip.tif'

# Open the TIF file with rasterio
with rasterio.open(file_path) as src:
    # Read the data from the TIF file
    img = src.read()
    
    # Use rasterio's 'show' function to display the image
    show(img)
    
    # Optional: you can customize the plot with matplotlib functions
    plt.title('20230110_042730_39_2413_3B_AnalyticMS_SR_clip')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Display the plot
    plt.show()

# %%
