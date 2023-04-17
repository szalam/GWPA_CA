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
well_src       = 'GAMA'          # options: UCD, GAMA
rad_buffer     = 5              # well buffer radius in miles
buffer_flag    = 0              # flag 1: use existing buffer shapefile; 0: create buffer
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
data = dp.get_luids()
data_names = dp.get_luid_names()
lu_filnames = dp.lu_cdl_filname()
# 6,7, 9, 10, 11, 12, 14, 16, 17,18, 21

for yr in tqdm(range(2007, 2022)):
    # Open the TIF file using rasterio
    with rasterio.open(config.data_raw / f'USDA_CDL/{lu_filnames[yr][0]}') as src:
        landuse = src.read(1)

        # Get the transform information to convert the GeoDataFrame
        transform = src.transform
        

    # Open the shapefile using geopandas
    wells = gdf_wellbuffer.copy() #gpd.read_file('path/to/geodataframe.shp')

    # Reproject the GeoDataFrame to match the TIF file
    wells = wells.to_crs(src.crs)

    # Create an empty dictionary to store the results
    results = {}

    #===========================================================
    # Create an empty dictionary to store the results
    # Create an empty dictionary to store the results
    results = {'well_id': [], 'Row_crops': [], 'Small_grains': [], 'Truck_nursery_berry': [], 'Citrus_subtropical': [], 'Field_crops': [], 'Vineyards': [],'Grain_hay': [], 'Deciduous_fruits_nuts': [], 'Rice': [], 'Cotton': [], 'Other_crops': [],'Idle': [] , 'All_crop':[]}

    # Loop through each well and calculate the area for each category
    n_count = 0
    # filter out all warnings
    warnings.filterwarnings("ignore")
    for well in wells.itertuples():
        n_count = n_count+1
        print(n_count)
        mask = rasterio.features.geometry_mask([well.geometry], transform=src.transform, out_shape=landuse.shape, invert=True)
        extracted_landuse = np.extract(mask, landuse)


        Row_crops_area = np.count_nonzero(np.isin(extracted_landuse, data[6])) * 900 # 900 is 30*30
        Small_grains_area = np.count_nonzero(np.isin(extracted_landuse, data[7])) * 900 # 900 is 30*30
        Truck_nursery_berry_area = np.count_nonzero(np.isin(extracted_landuse, data[9])) * 900 # 900 is 30*30
        Citrus_subtropical_area = np.count_nonzero(np.isin(extracted_landuse, data[10])) * 900 # 900 is 30*30
        Field_crops_area = np.count_nonzero(np.isin(extracted_landuse, data[11])) * 900 # 900 is 30*30
        Vineyards_area = np.count_nonzero(np.isin(extracted_landuse, data[12])) * 900 # 900 is 30*30
        Grain_hay_area = np.count_nonzero(np.isin(extracted_landuse, data[14])) * 900 # 900 is 30*30
        Fruits_nuts = np.count_nonzero(np.isin(extracted_landuse, data[16])) * 900 # 900 is 30*30
        Rice = np.count_nonzero(np.isin(extracted_landuse, data[17])) * 900 # 900 is 30*30
        Cotton = np.count_nonzero(np.isin(extracted_landuse, data[18])) * 900 # 900 is 30*30
        Other_crops = np.count_nonzero(np.isin(extracted_landuse, data[21])) * 900 # 900 is 30*30
        Idle = np.count_nonzero(np.isin(extracted_landuse, data[8])) * 900 # 900 is 30*30
        All_crop = np.count_nonzero(np.isin(extracted_landuse, data[23])) * 900 # 900 is 30*30
        
        results['well_id'].append(well.well_id)
        results['Row_crops'].append(Row_crops_area)
        results['Small_grains'].append(Small_grains_area)
        results['Truck_nursery_berry'].append(Truck_nursery_berry_area)
        results['Citrus_subtropical'].append(Citrus_subtropical_area)
        results['Field_crops'].append(Field_crops_area)
        results['Vineyards'].append(Vineyards_area)
        results['Grain_hay'].append(Grain_hay_area)
        results['Deciduous_fruits_nuts'].append(Fruits_nuts)
        results['Rice'].append(Rice)
        results['Cotton'].append(Cotton)
        results['Other_crops'].append(Other_crops)
        results['Idle'].append(Idle)
        results['All_crop'].append(All_crop)

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(results)

    # Export the DataFrame to a CSV file
    df.to_csv(config.data_processed / f"CDL/cdl_at_buffers/{well_src}/CDL_{yr}.csv")

# %%
