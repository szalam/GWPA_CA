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
from scipy import stats
import seaborn as sns
import matplotlib as mpl

cond_type_used = 'Conductivity_lyrs_1'

# Read dataset
df = pd.read_csv(config.data_processed / "Dataset_processed.csv")
df = df[df.well_data_source == 'GAMA']
df = df[df.measurement_count > 1]
# df = df[df.city_inside_outside == 'outside_city']
# df = df[df.city_inside_outside == 'inside_city']
df = df[[f'{cond_type_used}','area_wt_sagbi']]
df = df.dropna()

#%%
plt.scatter(df[cond_type_used], df.area_wt_sagbi, s = 1.5, c = 'red')
plt.ylim(0 ,100)
plt.xlabel('Conductivity')
plt.ylabel('SAGBI Rating')
# %%

#================= Boxplot =======================

# Bin Conductivity into intervals of .05
df['Conductivity_binned'] = pd.cut(df[f'{cond_type_used}'], np.arange(0, df[f'{cond_type_used}'].max()+.1,.1))

# Increase the font size
# mpl.rcParams.update({'font.size': 14})

# Create a box and whisker plot using Seaborn
sns.boxplot(x='Conductivity_binned', y='area_wt_sagbi', data=df, width=0.5)

# Add x and y labels
plt.xlabel('Conductivity', fontsize = 17)
plt.ylabel('SAGBI Rating', fontsize =17)
plt.tick_params(axis='both', which='major', labelsize=14)

# Set y-axis limit to 0 to 100
plt.ylim(0, 100)

# Rotate x tick labels for better readability
plt.xticks(rotation=90)

# Show plot
plt.show()
# %%
