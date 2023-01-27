#================================================================================
# This script analyzies the relationship between animal population and nitrate
# for wells when they have almost similar conductivity
#================================================================================
#%%
import sys
sys.path.insert(0,'src')
sys.path.insert(0, 'src/data')

import pandas as pd
import config
import numpy as np
import matplotlib.pyplot as plt 
import ppfun as dp


# Read dataset
df = pd.read_csv(config.data_processed / "Dataset_processed.csv")
df = df[df.well_data_source == 'UCD']
df = df[df.measurement_count>4]
df = df[['well_id','Conductivity','mean_nitrate','area_wt_sagbi', 'Cafo_Population_5miles','Average_ag_area','change_per_year','total_ag_2010','APPROXIMATE LATITUDE', 'APPROXIMATE LONGITUDE','Conductivity_depthwtd_lyr1','Conductivity_depthwtd_lyr2','Conductivity_depthwtd_lyr3','Conductivity_depthwtd_lyr10','Conductivity_depthwtd_lyr19']]

# df = df.head(1000)

# Create bins for area_wt_sagbi column
bins = [i for i in range(0, 110, 10)]

# Create labels for each bin
labels = [f'{i} - {i+10}' for i in range(0, 100, 10)]

# Add a new column to the dataframe with the binned values
df['area_wt_sagbi_binned'] = pd.cut(df['area_wt_sagbi'], bins=bins, labels=labels)

# Group the dataframe by the binned column
grouped = df.groupby('area_wt_sagbi_binned')

#%%
min_lim = 0
max_lim = 1000
bins = np.linspace(min_lim, max_lim, 21)
# Iterate through each group and plot the histogram for the Nitrate column
for group, data in grouped:
    plt.hist(data['mean_nitrate'], bins=bins)
    plt.xlabel("Nitrate [mg/l]")
    plt.xlim(min_lim ,max_lim)
    plt.title(f"SAGBI: {group}")
    plt.show()
# %%
min_lim = 0
max_lim = .5
bins = np.linspace(min_lim, max_lim)
# Iterate through each group and plot the histogram for the Nitrate column
for group, data in grouped:
    plt.hist(data['Conductivity'], bins=bins)
    plt.xlabel("Conductivity")
    plt.xlim(min_lim ,max_lim)
    plt.title(f"SAGBI: {group}")
    plt.show()
# %%
plt.hist(df['Conductivity'], bins=300)
plt.xlabel("Conductivity")
plt.xlim(0 ,0.8)
plt.show()
# %%
# %%
plt.hist(df['Conductivity_depthwtd_lyr1'], bins=300)
plt.xlabel("Conductivity")
# plt.xlim(0 ,0.8)
plt.show()
# %%
