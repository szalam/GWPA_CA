#%%
import sys
sys.path.insert(0,'src')
import config
import pandas as pd
import ppfun as dp
import geopandas as gpd
from shapely.geometry import Point

# CAFO dating data 
cafo_dts = pd.read_csv(config.data_processed / "CAFO_dating/CA_final_with_dates.csv")
# cafo_dts_gdf = gpd.GeoDataFrame(cafo_dts, geometry=gpd.points_from_xy(cafo_dts.longitude, cafo_dts.latitude))

# Convert latitude and longitude columns to shapely Points
geometry = [Point(xy) for xy in zip(cafo_dts['longitude'], cafo_dts['latitude'])]

# Create a GeoDataFrame
crs = {'init': 'epsg:4326'} # CRS stands for Coordinate Reference System, this line specify the system
cafo_dts_gdf = gpd.GeoDataFrame(cafo_dts, crs=crs, geometry=geometry)
# cafo_dts_gdf.to_file(config.data_processed / 'kml' / f"CAFO_shape.shp", driver='ESRI Shapefile')


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

wellbuff_cafo_intersect = gpd.overlay(cafo_dts_gdf, gdf_wellbuffer,how='intersection')
gdf_cafopop_in_buff = wellbuff_cafo_intersect.groupby(["well_id"]).Cafo_Population.sum().reset_index()

# Assign column names to the variables 'well_id' and 'CAFO_population'
gdf_cafopop_in_buff.columns = ['well_id', f'CAFO_population_{rad_buffer}miles']


# Export CSV with cafo population in well buffer
gdf_cafopop_in_buff.to_csv(config.data_processed / f"cafo_pop_wellbuffer/Cafopop_wellsrc_{well_src}_rad_{rad_buffer}mile.csv")

# %%
