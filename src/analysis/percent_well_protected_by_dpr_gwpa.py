# This script uses well data and DPR groundwater protection area
# to check percent wells protected by gwpa

#%%
import sys
sys.path.insert(0,'src')
sys.path.insert(0, 'src/data')
import config
import ppfun as dp
import numpy as np
import pandas as pd
import geopandas as gpd

columns_to_keep = ['well_id','Conductivity','measurement_count','mean_nitrate','area_wt_sagbi', 'Cafo_Population_5miles','Average_ag_area','change_per_year','total_ag_2010','APPROXIMATE LATITUDE', 'APPROXIMATE LONGITUDE','city_inside_outside']

# Read well dataset
df = pd.read_csv(config.data_processed / "Dataset_processed.csv")
df = df[df.well_data_source == 'UCD']
df = df[columns_to_keep]
df = df[df.mean_nitrate>=10]
df = df[df.measurement_count > 4]
# df = df[df.city_inside_outside == 'outside_city']
# df = df[df.city_inside_outside == 'inside_city']

well_pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['APPROXIMATE LONGITUDE'], df['APPROXIMATE LATITUDE']))

# Read shapefiles and adjust coordinates
cv = dp.get_region(config.shapefile_dir   / "cv.shp")
gwpa = dp.point_clip((config.shapefile_dir   / "ca_statewide_gwpa/CA_Statewide_GWPAs.shp"), reg = cv)
gwpa = gwpa.to_crs({'init': 'epsg:4326'})
cv = cv.to_crs({'init': 'epsg:4326'})
well_pts = well_pts.set_crs({'init': 'epsg:4326'})

wells_in_cv = gpd.clip(well_pts, cv) #clip wpa using kw shape

#%%
# perform spatial join
wells_in_gwpa = gpd.sjoin(wells_in_cv, gwpa, how="left", op='within')
wells_in_gwpa["gwpa_inout"] = np.where(wells_in_gwpa["index_right"].isnull(), "outside_gwpa", "inside_gwpa")
wells_in_gwpa_back = wells_in_gwpa.copy()

# keep required columns only
wells_in_gwpa = wells_in_gwpa[['well_id', 'gwpa_inout','mean_nitrate','Conductivity']]
# %%
well_outside_gwpa_count = wells_in_gwpa['gwpa_inout'].value_counts()['outside_gwpa']
well_inside_gwpa_count = wells_in_gwpa['gwpa_inout'].value_counts()['inside_gwpa']
# %%
print(f'Percent of wells protected by GWPA for nitrate pollution = {round(well_inside_gwpa_count/wells_in_gwpa.shape[0]*100,2)}')
# %%
# wells_in_gwpa_back = wells_in_gwpa_back[wells_in_gwpa_back.gwpa_inout == 'outside_gwpa' ]
# wells_in_gwpa_back.to_file(config.data_processed / 'kml' / "wells_outside_gwpa_N_above10_outsideCity.shp", driver='ESRI Shapefile')
# %%
