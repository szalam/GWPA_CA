#%%
import os
import sys
sys.path.insert(0,'src')
import config
import pandas as pd
import numpy as np

gama = pd.read_csv(f'{config.data_processed}/well_stats/gamanitrate_latest_stats.csv')
# %%
gama.columns
# %%
gama = gama[['well_id','APPROXIMATE LATITUDE',
       'APPROXIMATE LONGITUDE','mean_nitrate','measurement_count',
       'well_type']]
# %%
gama.columns
# %%
gama.to_csv(config.data_raw / 'shapefile/GAMAlatest_wells.csv')
# %%
