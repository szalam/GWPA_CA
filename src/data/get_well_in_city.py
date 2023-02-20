#%%
import sys
sys.path.insert(0,'src')
import pandas as pd
import config
import ppfun as dp
import geopandas as gpd
import numpy as np
from shapely.geometry import Point


# Function
def get_city_well_inout(df,city):
    # Convert latitude and longitude columns to shapely Points
    geometry = [Point(xy) for xy in zip(df['APPROXIMATE LONGITUDE'], df['APPROXIMATE LATITUDE'])]
    # Create a GeoDataFrame
    crs = {'init': 'epsg:4326'} # CRS stands for Coordinate Reference System, this line specify the system
    gdf2 = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)

    city = city.to_crs({'init': 'epsg:4326'})
    # perform spatial join
    points_with_city = gpd.sjoin(gdf2, city, how="left", op='within')
    points_with_city["city_inside_outside"] = np.where(points_with_city["index_right"].isnull(), "outside_city", "inside_city")
    # Keep only well_id and city_inside_outside columns in points_with_city
    points_with_city = points_with_city[['well_id', 'city_inside_outside']]
    
    # Drop duplicate and keep the first one
    points_with_city.drop_duplicates(subset = ['well_id'], keep = 'first', inplace = True) 
    
    return points_with_city

# read in city shapefile
city = gpd.read_file(config.data_raw / "shapefile/City_Boundaries/City_Boundaries.shp")

#======================
# GAMA data
#======================
# read gama excel file
file_polut = config.data_gama_all / 'CENTRALVALLEY_NO3N_GAMA.csv'
df_gama= dp.get_polut_df(file_sel = file_polut)
df_gama.rename(columns = {'GM_WELL_ID':'well_id', 'GM_LATITUDE':'APPROXIMATE LATITUDE', 'GM_LONGITUDE':'APPROXIMATE LONGITUDE', 'GM_CHEMICAL_VVL': 'CHEMICAL', 'GM_RESULT': 'RESULT','GM_WELL_CATEGORY':'well_type','GM_SAMP_COLLECTION_DATE':'DATE'}, inplace = True)
df_gama['DATE']= pd.to_datetime(df_gama['DATE'])

# df_gama = pd.read_excel(config.data_gama / 'TULARE_NO3N.xlsx',engine='openpyxl')
# df_gama.rename(columns = {'GM_WELL_ID':'well_id', 'GM_LATITUDE':'APPROXIMATE LATITUDE', 'GM_LONGITUDE':'APPROXIMATE LONGITUDE', 'GM_CHEMICAL_VVL': 'CHEMICAL', 'GM_RESULT': 'RESULT','GM_WELL_CATEGORY':'well_type','GM_SAMP_COLLECTION_DATE':'DATE'}, inplace = True)

points_with_city_gama = get_city_well_inout(df_gama, city)
#======================
# UCD data
#======================
# file location
file_polut = config.data_raw / "nitrate_data/UCDNitrateData.csv"
# read data
df_ucd = dp.get_polut_df(file_sel = file_polut)
df_ucd = df_ucd.rename(columns={'WELL ID': 'well_id','DATASET_CAT': 'well_type'}) 
# Convert the date column to a datetime object
df_ucd["date"] = pd.to_datetime(df_ucd["DATE"])

points_with_city_ucd = get_city_well_inout(df_ucd, city)
#%%
# Concatenate both dataframes one above the other
points_with_city_combine = pd.concat([points_with_city_gama, points_with_city_ucd])
points_with_city_combine.to_csv(config.data_processed / "well_inout_city/well_inout_city.csv", index=False)
# %%
