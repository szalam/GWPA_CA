#%%
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, 'src')
import config

#%%
res_var = [
    'Resistivity_lyrs_9_rad_2_miles'
]

#%%

# Constants
rad_well = 2
gama_old_new = 2 # 1: earlier version, 2: latest version of GAMA
all_dom_flag = 1 # 1: All, 2: Domestic
if all_dom_flag == 2:
    well_type_select = {1: 'Domestic', 2: 'DOMESTIC'}.get(gama_old_new) 
else:
    well_type_select = 'All'

# Read dataset
def load_data(version):
    """Load data based on version"""
    filename = "Dataset_processed_GAMAlatest.csv" if version == 2 else "Dataset_processed.csv"
    df = pd.read_csv(config.data_processed / filename)
    return df

def filter_data(df, well_type,all_dom_flag):
    """Filter"""
    exclude_subregions = [14, 15, 10, 19, 18, 9, 6]
    if all_dom_flag == 2:
        df = df[df.well_type ==  well_type] 
    df = df[(df[f'thickness_abovCond_{round(.1*100)}_lyrs_9_rad_2miles'] <= 31) | (~df['SubRegion'].isin(exclude_subregions))]
    return df

# Load and process data
df_main = load_data(gama_old_new)
df = df_main[df_main.well_data_source == 'GAMA'].copy()

df['well_type_encoded'] = pd.factorize(df['well_type'])[0]
df['well_type_encoded'] = df['well_type_encoded'].where(df['well_type'].notna(), df['well_type'])

# separate wells inside cv
well_cv = pd.read_csv(config.data_processed / 'wells_inside_CV_GAMAlatest.csv',index_col=False)
# Assuming df is your dataframe with all wells
df_cv = df[df['well_id'].isin(well_cv['well_id'])]

df = filter_data(df_cv, well_type_select,all_dom_flag)
#%%
df = df.drop(['well_id', 'well_data_source','start_date', 'end_date'], axis=1)

columns_to_keep = ['mean_nitrate','SubRegion']

df2 = df[columns_to_keep]
df2 = df2.dropna()
# %%
import scipy.stats as stats

# First, we will create a list of mean_nitrate for each SubRegion
subregions = df2['SubRegion'].unique()
nitrate_values = [df2['mean_nitrate'][df2['SubRegion'] == region] for region in subregions]

# Then, we use scipy's f_oneway function to conduct the ANOVA test
F, p = stats.f_oneway(*nitrate_values)

print(f"F statistic: {F}")
print(f"P-value: {p}")

#%%
df_n_sub = df2.copy()
df_n_sub.to_csv('/Users/szalam/Main/00_Research_projects/mean_nit_sub.csv')
#%%
sc_sub = [1,2,3,5,6,7,8,9]
sj_sub  = [10, 11,12,13]
tul_sub  = [14, 15,16,17,18,19,20,21]
df_filt = df2[df2['SubRegion'].isin(tul_sub)]

# %%
import statsmodels.stats.multicomp as multi

# Perform Tukey's HSD test
mc = multi.MultiComparison(df_filt['mean_nitrate'], df_filt['SubRegion'])
result = mc.tukeyhsd()

print(result)
print(mc.groupsunique)
# %%
# Plot the result
result.plot_simultaneous(xlabel="Nitrate Concentration Difference")

# %%
# First, pivot your dataframe so each subregion becomes a column
df_pivot = df_filt.pivot(columns='SubRegion', values='mean_nitrate')

# Calculate pairwise correlation
correlation_matrix = df_pivot.corr()

print(correlation_matrix)
# %%
