#%%
import rasterio
import sys
import os
sys.path.insert(0,'src')
import warnings
import pandas as pd
# from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import config
import ppfun as dp
import numpy as np

from rasterio.features import geometry_mask
import geopandas as gpd
from tqdm import tqdm
from shapely.geometry import box


#======================== Get Well Buffer ================================
# file location
file_polut = config.data_raw / "nitrate_data/UCDNitrateData.csv"
file_aem = config.data_processed / 'AEM'

# Read region boundary
cv = dp.get_region(config.shapefile_dir   / "cv.shp")

#==================== User Input requred ==========================================
well_src       = 'GAMA'         # options: UCD, GAMA
rad_buffer     = 2              # well buffer radius in miles
buffer_flag    = 0              # flag 1: use existing buffer shapefile; 0: create buffer
file_name      = 'N_total'      # ProbDOpt5ppm_Deep,N_total,ProbDOpt5ppm_Shallow,ProbMn50ppb_Deep, ProbMn50ppb_Shallow
#==================================================================================

#=========================== Import water quality data ==========================
if well_src == 'GAMA':
    # read gama excel file
    # df = pd.read_excel(config.data_gama / 'TULARE_NO3N.xlsx',engine='openpyxl')
    # df.rename(columns = {'GM_WELL_ID':'WELL ID', 'GM_LATITUDE':'APPROXIMATE LATITUDE', 'GM_LONGITUDE':'APPROXIMATE LONGITUDE', 'GM_CHEMICAL_VVL': 'CHEMICAL', 'GM_RESULT': 'RESULT','GM_WELL_CATEGORY':'DATASET_CAT','GM_SAMP_COLLECTION_DATE':'DATE'}, inplace = True)
    # df['DATE']= pd.to_datetime(df['DATE'])
    file_polut = config.data_gama_all / 'CENTRALVALLEY_NO3N_GAMA.csv'
    df = dp.get_polut_df(file_sel = file_polut)
    df.rename(columns = {'GM_WELL_ID':'WELL ID', 'GM_LATITUDE':'APPROXIMATE LATITUDE', 'GM_LONGITUDE':'APPROXIMATE LONGITUDE', 'GM_CHEMICAL_VVL': 'CHEMICAL', 'GM_RESULT': 'RESULT','GM_WELL_CATEGORY':'well_type','GM_SAMP_COLLECTION_DATE':'DATE'}, inplace = True)
    df['DATE']= pd.to_datetime(df['DATE'])

if well_src == 'UCD':
    # file location
    file_polut = config.data_raw / "nitrate_data/UCDNitrateData.csv"

    # Read nitrate data
    df = dp.get_polut_df(file_sel = file_polut)

# Group the DataFrame by the 'WELL ID' column and take the first values for other selected columns
df = df.groupby('WELL ID')['APPROXIMATE LATITUDE', 'APPROXIMATE LONGITUDE',].apply(lambda x: x.iloc[0]).reset_index()

#======================== Clip around boundary ========================
# converting point resistivity data to polygon mask
# Creating buffer around wells
gdf_wellbuffer_all = dp.get_well_buffer_shape(df,rad_buffer = rad_buffer) 

# Perform  clip on the well data based on selected boundary
gdf_wellbuffer = gpd.sjoin(gdf_wellbuffer_all, cv, how='inner', op='within')
gdf_wellbuffer = gdf_wellbuffer[['well_id', 'lat', 'lon', 'geometry']]
# # Convert the CRS back to WGS 84
gdf_wellbuffer = gdf_wellbuffer.to_crs({'init': 'epsg:5070'}) # EPSG:5070 - NAD83 / Conus Albers; similar to CDL

gdf_wellbuffer = gdf_wellbuffer
# 6,7, 9, 10, 11, 12, 14, 16, 17,18, 21

#%%

# Open the TIF file using rasterio
with rasterio.open(config.data_processed / f'redox_Ninput_katetal/{file_name}.tif') as src:
    var = src.read(1)

    # Get the transform information to convert the GeoDataFrame
    transform = src.transform


# Open the shapefile using geopandas
wells = gdf_wellbuffer.copy() #gpd.read_file('path/to/geodataframe.shp')

# Reproject the GeoDataFrame to match the TIF file
wells = wells.to_crs(src.crs)

# Create an empty dictionary to store the results
results = []

#===========================================================
# Assuming 'wells' is a GeoDataFrame containing well geometries and 'var' is the raster data
results = {'well_id': [], 'mean_value': []}

# Loop through each well and calculate the area for each category
n_count = 0
warnings.filterwarnings("ignore")
for well in wells.itertuples():
    n_count += 1
    mask = rasterio.features.geometry_mask([well.geometry], transform=src.transform, out_shape=var.shape, invert=True)
    masked_var = np.ma.masked_array(var, mask=~mask)
    results['well_id'].append(well.well_id)  # Assuming the well_id is stored in an 'id' attribute
    results['mean_value'].append(masked_var.mean())

# Create a DataFrame from the results dictionary
df = pd.DataFrame(results)
#%%
# Export the DataFrame to a CSV file
df.to_csv(config.data_processed / f"redox_Ninput_katetal/exported_csv_redox_Ninput/{file_name}.csv")

# %%
