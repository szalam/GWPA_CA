#%%
import sys
sys.path.insert(0,'src')
import config 

import pandas as pd
#%%
# Read dataset
df = pd.read_csv(config.data_processed / "Dataset_processed.csv")
df = df[df.well_data_source == 'GAMA']
df = df[['APPROXIMATE LATITUDE','APPROXIMATE LONGITUDE','mean_nitrate']]
# %%
df = df.rename(columns={'APPROXIMATE LATITUDE': 'latitude',
               'APPROXIMATE LONGITUDE': 'longitude'})
# %%
df.to_csv(config.data_processed / 'nitrate_latlon.csv')
# %%
