#%%
import sys
sys.path.insert(0,'src')
import config
import pandas as pd
import ppfun as dp
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

# Read region boundary
cv = dp.get_region(config.shapefile_dir   / "cv.shp")

#==================== User Input requred ==========================================
well_src       = 'GAMA'          # options: UCD, GAMA
rad_buffer     = 2              # well buffer radius in miles
buffer_flag    = 0              # flag 1: use existing buffer shapefile; 0: create buffer
#==================================================================================

#=========================== Import water quality data ==========================
if well_src == 'GAMA':
    # read gama excel file
    df = pd.read_excel(config.data_gama / 'TULARE_NO3N.xlsx',engine='openpyxl')
    df.rename(columns = {'GM_WELL_ID':'WELL ID', 'GM_LATITUDE':'APPROXIMATE LATITUDE', 'GM_LONGITUDE':'APPROXIMATE LONGITUDE', 'GM_CHEMICAL_VVL': 'CHEMICAL', 'GM_RESULT': 'RESULT','GM_WELL_CATEGORY':'DATASET_CAT','GM_SAMP_COLLECTION_DATE':'DATE'}, inplace = True)
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


# Groundwater depth data read
gwdp = np.load(config.data_processed / 'GWdepth_interpolated' / 'GWdepth_spring.npy')
gwdp_x = np.load(config.data_processed / 'GWdepth_interpolated' / 'X_cv_gwdepth.npy')
gwdp_y = np.load(config.data_processed / 'GWdepth_interpolated' / 'Y_cv_gwdepth.npy')

gwd_gdf = dp.gwdep_array_gdf(gwdp,gwdp_x,gwdp_y)


 # spatial join of wq data and gw depth data
gwd_well_buff = gpd.sjoin_nearest(gdf_wellbuffer, gwd_gdf)

# # Get the intersections between the AEM data and the well buffer
# gwd_well_buff = gpd.overlay(gwd_gdf,gdf_wellbuffer, how='intersection')


gwd_well_buff = gwd_well_buff.drop(['geometry'], axis=1)

# Export CSV with cafo population in well buffer
gwd_well_buff.to_csv(config.data_processed / f"gwdepth_wellbuff/GWDepth_wellsrc_{well_src}_rad_{rad_buffer}mile.csv")


# # %%
# import matplotlib.pyplot as plt
# # Create a new figure
# fig, ax = plt.subplots()
# # Plot the first GeoDataFrame
# gwd_gdf.plot(ax=ax, color='blue')
# # Plot the second GeoDataFrame
# gdf_wellbuffer.plot(ax=ax, color='red')
# %%
