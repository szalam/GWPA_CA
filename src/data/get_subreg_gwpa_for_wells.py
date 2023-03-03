#%%
import sys
sys.path.insert(0,'src')
import pandas as pd
import config
import ppfun as dp
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import get_infodata as idt


#%%
# Function
def get_subreg_well(df,cv_subreg):
    # Convert latitude and longitude columns to shapely Points
    geometry = [Point(xy) for xy in zip(df['APPROXIMATE LONGITUDE'], df['APPROXIMATE LATITUDE'])]
    # Create a GeoDataFrame
    crs = {'init': 'epsg:4326'} # CRS stands for Coordinate Reference System, this line specify the system
    gdf2 = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)

    cv_subreg = cv_subreg.to_crs({'init': 'epsg:4326'})
    gdf3 = gpd.sjoin(gdf2, cv_subreg, op="within")
    df_result = gdf3[['well_id', 'SubRegion', 'HR']]

    return df_result

# Function
def get_gwpa_well(df,gwpa):
    # Convert latitude and longitude columns to shapely Points
    geometry = [Point(xy) for xy in zip(df['APPROXIMATE LONGITUDE'], df['APPROXIMATE LATITUDE'])]
    # Create a GeoDataFrame
    crs = {'init': 'epsg:4326'} # CRS stands for Coordinate Reference System, this line specify the system
    gdf2 = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)

    gwpa = gwpa.to_crs({'init': 'epsg:4326'})
    gdf3 = gpd.sjoin(gdf2, gwpa, op="within")
    df_result = gdf3[['well_id', 'GWPAType', 'GWPA_Year','SECTION']]

    return df_result

# read in city shapefile
cv_subreg = idt.get_region(reg_sel = 'cv_subreg')

#%%
#======================
# GAMA data
#======================
# read gama excel file
file_polut = config.data_gama_all / 'CENTRALVALLEY_NO3N_GAMA.csv'
df_gama= dp.get_polut_df(file_sel = file_polut)
df_gama.rename(columns = {'GM_WELL_ID':'well_id', 'GM_LATITUDE':'APPROXIMATE LATITUDE', 'GM_LONGITUDE':'APPROXIMATE LONGITUDE', 'GM_CHEMICAL_VVL': 'CHEMICAL', 'GM_RESULT': 'RESULT','GM_WELL_CATEGORY':'well_type','GM_SAMP_COLLECTION_DATE':'DATE'}, inplace = True)
df_gama = df_gama[['well_id', 'APPROXIMATE LATITUDE', 'APPROXIMATE LONGITUDE']]
df_gama = df_gama.drop_duplicates()

# df_gama = pd.read_excel(config.data_gama / 'TULARE_NO3N.xlsx',engine='openpyxl')
# df_gama.rename(columns = {'GM_WELL_ID':'well_id', 'GM_LATITUDE':'APPROXIMATE LATITUDE', 'GM_LONGITUDE':'APPROXIMATE LONGITUDE', 'GM_CHEMICAL_VVL': 'CHEMICAL', 'GM_RESULT': 'RESULT','GM_WELL_CATEGORY':'well_type','GM_SAMP_COLLECTION_DATE':'DATE'}, inplace = True)

points_with_subreg_gama = get_subreg_well(df_gama, cv_subreg)
#======================
# UCD data
#======================
# file location
file_polut = config.data_raw / "nitrate_data/UCDNitrateData.csv"
# read data
df_ucd = dp.get_polut_df(file_sel = file_polut)
df_ucd = df_ucd.rename(columns={'WELL ID': 'well_id','DATASET_CAT': 'well_type'}) 
df_ucd = df_ucd[['well_id', 'APPROXIMATE LATITUDE', 'APPROXIMATE LONGITUDE']]
df_ucd = df_ucd.drop_duplicates()

points_with_subreg_ucd = get_subreg_well(df_ucd, cv_subreg)
#%%
# Concatenate both dataframes one above the other
points_with_subreg_combine = pd.concat([points_with_subreg_gama, points_with_subreg_ucd])
points_with_subreg_combine.to_csv(config.data_processed /"well_in_subregions.csv", index=False)

# %%
gwpa = dp.get_region(config.shapefile_dir   / "ca_statewide_gwpa/CA_Statewide_GWPAs.shp")
points_with_gwpa_ucd = get_gwpa_well(df_ucd, gwpa)
points_with_gwpa_gama = get_gwpa_well(df_gama, gwpa)
points_with_gwpa_combine = pd.concat([points_with_gwpa_ucd, points_with_gwpa_gama])
points_with_gwpa_combine.to_csv(config.data_processed /"well_in_gwpa.csv", index=False)

# %%
