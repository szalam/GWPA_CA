#%%
import sys
sys.path.insert(0,'src')
import config 

import matplotlib.pyplot as plt
import contextily as ctx
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
#%%
# Read dataset
df = pd.read_csv(config.data_processed / "Dataset_processed.csv")
df = df[df.well_data_source == 'GAMA']
df = df[['APPROXIMATE LATITUDE','APPROXIMATE LONGITUDE','mean_nitrate']]
# %%
df = df.rename(columns={'APPROXIMATE LATITUDE': 'lat',
               'APPROXIMATE LONGITUDE': 'lon'})
# %%
df.head(2)
# %%
geometry = [Point(xy) for xy in zip(df.lon, df.lat)]
gdf = gpd.GeoDataFrame(df, geometry=geometry)

#%%# Step 4: Get the zipcodes
gdf_zipcodes = gpd.read_file('https://www2.census.gov/geo/tiger/TIGER2018/ZCTA5/tl_2018_us_zcta510.zip')

# Step 5: Spatial join
gdf_zipcodes_nitrate = gpd.sjoin(gdf, gdf_zipcodes, op='within', how='left')

# Step 6: Groupby
gdf_zipcodes_mean = gdf_zipcodes_nitrate.groupby('ZCTA5CE10').mean()

#%%
fig, ax = plt.subplots(figsize=(12, 8))
gdf_zipcodes_mean.plot(column='mean_nitrate', ax=ax, legend=True, cmap='YlGnBu')
ax.axis('off')
ctx.add_basemap(ax, crs=gdf_zipcodes_mean.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)
plt.show()
# %%
