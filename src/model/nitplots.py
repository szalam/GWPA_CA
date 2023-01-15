#%%
import sys
sys.path.insert(0,'src')

import pandas as pd
import config
import matplotlib.pyplot as plt 

from visualize import vismod

# Read dataset
df = pd.read_csv(config.data_processed / "Dataset_processed.csv")
df['nitrate_increase'] = df['mean_concentration_2015-2022']- df['mean_concentration_2005-2010']

# Separate data from GAMA and UCD
dfgama = df[df.well_data_source == 'GAMA']
dfucd = df[df.well_data_source == 'UCD']
#%%
mod = vismod(df = df)
modgama = vismod(df = dfgama)
moducd = vismod(df = dfucd)

# 'mean_nitrate', 'median_nitrate', 'max_nitrate', 'min_nitrate',
# 'measurement_count', 'mean_concentration_2015-2022',
# 'mean_concentration_2010-2015', 'mean_concentration_2005-2010',
# 'mean_concentration_2000-2005', 'mean_concentration_2000-2022',
# 'mean_concentration_2010-2022', 'mean_concentration_2007-2009',
# 'mean_concentration_2012-2015', 'mean_concentration_2019-2021',
# 'mean_concentration_2017-2018'
# %%
mod.get_scatter_aem_polut(xcolm_name='Conductivity', ycolm_name='mean_nitrate', 
        yaxis_lim = 1000, xaxis_lim = .8, YlabelC ='Mean Nitrate C.', XlabelC ='Conductivity' ,
        yunitylabel ='[mg/l]', xunitylabel =' ')
# %%
moducd.get_scatter_aem_polut(xcolm_name='Conductivity', ycolm_name='mean_concentration_2000-2022', 
        yaxis_lim = 1000, xaxis_lim = .8, YlabelC ='Mean Nitrate C.', XlabelC ='Conductivity' ,
        yunitylabel ='[mg/l]', xunitylabel =' ')
#%%
modgama.get_scatter_aem_polut(xcolm_name='Conductivity', ycolm_name='mean_concentration_2000-2022', 
        yaxis_lim = 100, xaxis_lim = .8, YlabelC ='Mean Nitrate C.', XlabelC ='Conductivity' ,
        yunitylabel ='[mg/l]', xunitylabel =' ')

# %%
mod.get_scatter_aem_polut(xcolm_name='Conductivity', ycolm_name='nitrate_increase', 
        yaxis_lim = 1000, xaxis_lim = .8, YlabelC ='Mean Nitrate C.', XlabelC ='Conductivity' ,
        yunitylabel ='[mg/l]', xunitylabel =' ')


# %%
mod.get_scatter_aem_polut(xcolm_name='area_wt_sagbi', ycolm_name='mean_concentration_2010-2022', 
        yaxis_lim = 150, xaxis_lim = None, YlabelC ='Mean Nitrate C.', XlabelC ='SAGBI Rating' ,
        yunitylabel ='[mg/l]', xunitylabel =' ')
# %%
# %%
mod.get_scatter_aem_polut(xcolm_name='mean_concentration_2000-2005', ycolm_name='mean_concentration_2010-2022', 
        yaxis_lim = 500, xaxis_lim = 500, YlabelC ='Mean Nitrate C.[2010-2022]', XlabelC ='Mean Nitrate C.[2000-2005]' ,
        yunitylabel =' ', xunitylabel =' ')
# %%
