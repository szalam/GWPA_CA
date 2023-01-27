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


data_src = 'GAMA'
# Read dataset
df = pd.read_csv(config.data_processed / "Dataset_processed.csv")
df = df[df.well_data_source == data_src]
df = pd.DataFrame(df, columns=['well_id','APPROXIMATE LONGITUDE','APPROXIMATE LATITUDE','mean_nitrate', 'Conductivity','total_obs'])
df = df[df.total_obs>4]
geo_df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['APPROXIMATE LONGITUDE'], df['APPROXIMATE LATITUDE']))

#%%
# geo_df.to_file(config.data_processed / 'kml' / f"mean_nitrate_{data_src}.shp", driver='ESRI Shapefile')

conductivity_threshold = .0
nitrate_abov = 0
condition_greater = 1 # options: 0 for condition below the conductivity threshold used above

mean_nitrate_df = geo_df[geo_df['mean_nitrate'] > nitrate_abov]
if condition_greater == 1:
    mean_nitrate_df = mean_nitrate_df[mean_nitrate_df['Conductivity'] > conductivity_threshold]
    mean_nitrate_df.to_file(config.data_processed / 'kml' / f"NAbov_{nitrate_abov}_CondAbov_{conductivity_threshold*100}_{data_src}.shp", driver='ESRI Shapefile')

if condition_greater == 0:
    mean_nitrate_df = mean_nitrate_df[mean_nitrate_df['Conductivity'] < conductivity_threshold]
    mean_nitrate_df.to_file(config.data_processed / 'kml' / f"NAbov_{nitrate_abov}_CondBelow_{conductivity_threshold*100}_{data_src}.shp", driver='ESRI Shapefile')



#%%
# mean_nitrate_df = geo_df[geo_df['mean_nitrate'] > 0]

# gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
# mean_nitrate_df.to_file(config.data_processed / 'kml' / f"mean_nitrate_{data_src}.kml", driver='KML')
