#%%
import sys
sys.path.insert(0,'src')
import numpy as np
import pandas as pd
import config
import rasterio
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import pyproj
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import Transformer
#%%

file_name = 'N_total'
# Define the custom Coordinate Reference System (CRS)
custom_crs = '+proj=aea +lat_0=23 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs'

# Open the TIF file using rasterio
file_path = config.data_processed / f'redox_Ninput_katetal/{file_name}.tif'
with rasterio.open(file_path) as src:
    # Set the custom CRS
    src_crs = src.crs
    dst_crs = rasterio.crs.CRS.from_string(custom_crs)

    # Calculate the transform
    transform, width, height = calculate_default_transform(src_crs, dst_crs, src.width, src.height, *src.bounds)
    
    # Create a new rasterio dataset to store the reprojected data
    with rasterio.open('reprojected.tif', 'w', driver='GTiff', height=height, width=width, count=src.count, dtype=src.dtypes[0], crs=dst_crs, transform=transform) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)

# Read the reprojected data
with rasterio.open('reprojected.tif') as reprojected_dst:
    var = reprojected_dst.read(1)

    # Get the transform information to convert the GeoDataFrame
    transform = reprojected_dst.transform

#%%
# Plot the reprojected data
plt.imshow(var, cmap='viridis', extent=reprojected_dst.bounds,vmin = 1000, vmax = 10000)
plt.colorbar(label='Values')
plt.title('Reprojected Data')
plt.xlabel('Easting')
plt.ylabel('Northing')
plt.show()


#%%
df_well = pd.read_csv(config.data_processed / "Dataset_processed.csv")

# Create a GeoDataFrame from the well information DataFrame
gdf_wells = gpd.GeoDataFrame(df_well, geometry=gpd.points_from_xy(df_well['APPROXIMATE LONGITUDE'], df_well['APPROXIMATE LATITUDE']),crs={'init': 'epsg:4326'})
gdf_wells = gdf_wells[gdf_wells.well_data_source == 'GAMA']
gdf_wells = gdf_wells[['well_id','APPROXIMATE LONGITUDE','APPROXIMATE LATITUDE','geometry']]
# gdf_wells.to_csv(config.data_processed / 'kml' / "GAMA_wells.csv")
#%%
# Step 1: Reproject gdf_wells to match the CRS of the reprojected raster (dst_crs)
gdf_wells_reprojected = gdf_wells.to_crs(dst_crs)

# Step 2: Plot the raster map (reprojected data)
plt.imshow(var, cmap='viridis', extent=reprojected_dst.bounds, vmin=1000, vmax=10000)
plt.colorbar(label='Values')

# Step 3: Overlay the points from gdf_wells_reprojected on top of the raster map
gdf_wells_reprojected.plot(ax=plt.gca(), color='red', markersize=3, marker='o')

# Add title and labels to the plot
plt.title('Reprojected Data with Wells')
plt.xlabel('Easting')
plt.ylabel('Northing')

# Show the plot
plt.show()

# %%
# Open the TIF file using rasterio
with rasterio.open(config.data_processed / f'redox_Ninput_katetal/{file_name}.tif') as src:
    var = src.read(1)

    # Get the transform information to convert the GeoDataFrame
    transform = src.transform

# %%
plt.imshow(var, cmap='viridis', extent=reprojected_dst.bounds, vmin=1000, vmax=10000)
plt.colorbar(label='Values')
# %%
wells = gdf_wells.to_crs(src.crs)
# %%

# Step 2: Plot the raster map (reprojected data)
plt.imshow(var, cmap='viridis', extent=reprojected_dst.bounds, vmin=1000, vmax=10000)
plt.colorbar(label='Values')

# Step 3: Overlay the points from gdf_wells_reprojected on top of the raster map
wells.plot(ax=plt.gca(), color='red', markersize=3, marker='o')

# Add title and labels to the plot
plt.title('Reprojected Data with Wells')
plt.xlabel('Easting')
plt.ylabel('Northing')

# %%
