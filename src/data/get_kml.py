#================================================================================
# This code reads dataframe and convert it to kml
#================================================================================
#%%
import sys
sys.path.insert(0,'src')
sys.path.insert(0, 'src/data')

import pandas as pd
import config
import matplotlib.pyplot as plt 
import ppfun as dp
import geopandas as gpd
from shapely.geometry import Point
import get_infodata as gi

#%%
data_src = 'UCD'
# Read dataset
df = pd.read_csv(config.data_processed / "Dataset_processed.csv")
df = df[df.well_data_source == data_src]
df = pd.DataFrame(df, columns=['well_id','APPROXIMATE LONGITUDE','APPROXIMATE LATITUDE','mean_nitrate', 'Conductivity','total_obs','Average_ag_area'])
df = df[df.total_obs>4]
# df = df[df.Average_ag_area>15000000]
geo_df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['APPROXIMATE LONGITUDE'], df['APPROXIMATE LATITUDE']))

#%%
# geo_df.to_file(config.data_processed / 'kml' / f"mean_nitrate_{data_src}.shp", driver='ESRI Shapefile')

conductivity_threshold = 0
nitrate_threshold = 10
nitrate_greater   = 1   # Options: 1=greater than, 0=smaller than
condition_greater = 1   # Options: 1=greater than, 0=smaller than

if nitrate_greater == 1:
    mean_nitrate_df = geo_df[geo_df['mean_nitrate'] > nitrate_threshold]
if nitrate_greater == 0:
    mean_nitrate_df = geo_df[geo_df['mean_nitrate'] < nitrate_threshold]
if condition_greater == 1:
    mean_nitrate_df = mean_nitrate_df[mean_nitrate_df['Conductivity'] > conductivity_threshold]
    mean_nitrate_df.to_file(config.data_processed / 'kml' / f"NAbov_{nitrate_threshold}_CondAbov_{conductivity_threshold*100}_{data_src}.shp", driver='ESRI Shapefile')

if condition_greater == 0:
    mean_nitrate_df = mean_nitrate_df[mean_nitrate_df['Conductivity'] < conductivity_threshold]
    mean_nitrate_df.to_file(config.data_processed / 'kml' / f"NAbov_{nitrate_threshold}_CondBelow_{conductivity_threshold*100}_{data_src}.shp", driver='ESRI Shapefile')


#%%
# export shape for wells having positive relationship with CAFO
df_well_cafo_pos = pd.read_csv(config.data_processed / "cafo_N_positive_relation_same_conductivity/wellids_have_cafo_positive_relations.csv")
df_well_cafo_pos = gpd.GeoDataFrame(df_well_cafo_pos, geometry=gpd.points_from_xy(df_well_cafo_pos['APPROXIMATE LONGITUDE'], df_well_cafo_pos['APPROXIMATE LATITUDE']))

df_well_cafo_pos.to_file(config.data_processed / 'kml' / f"well_cafo_pos_relations_{data_src}.shp", driver='ESRI Shapefile')
#%%
# mean_nitrate_df = geo_df[geo_df['mean_nitrate'] > 0]

# gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
# mean_nitrate_df.to_file(config.data_processed / 'kml' / f"mean_nitrate_{data_src}.kml", driver='KML')

#%%
gwpa = gi.get_basic_data(out_data='gwpa')
# %%
gwpa_leaching = gwpa[gwpa.GWPAType == 'Leaching']
gwpa_runoff = gwpa[gwpa.GWPAType == 'Runoff']
gwpa_leach_or_runoff = gwpa[gwpa.GWPAType == 'Leach/Runoff']

# %%

gwpa_leaching.to_file(config.data_processed / 'kml' / f"gwpa_leaching.shp", driver='ESRI Shapefile')
gwpa_runoff.to_file(config.data_processed / 'kml' / f"gwpa_runoff.shp", driver='ESRI Shapefile')
gwpa_leach_or_runoff.to_file(config.data_processed / 'kml' / f"gwpa_leach_or_runoff.shp", driver='ESRI Shapefile')

# %%
