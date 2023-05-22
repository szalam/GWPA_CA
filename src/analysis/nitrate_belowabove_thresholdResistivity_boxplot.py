#%%
import sys
sys.path.insert(0,'src')

import pandas as pd
import matplotlib.pyplot as plt
import config
import numpy as np
import seaborn as sns
from scipy.stats import mannwhitneyu
#%%
rad_well = 2
# Read dataset
df = pd.read_csv(config.data_processed / "Dataset_processed.csv")
df = df[df.well_data_source == 'GAMA']
df = df.drop(['well_id', 'well_data_source','start_date', 'end_date'], axis=1)

# df = df[df.measurement_count>5]
# df = df[df.well_type == 'Domestic']
# Remove high salinity regions
exclude_subregions = [14, 15, 10, 19,18, 9,6]
# # filter aemres to keep only the rows where Resistivity is >= 10 and SubRegion is not in the exclude_subregions list
df = df[(df[f'thickness_abovCond_{round(.1*100)}_lyrs_9_rad_{rad_well}miles'] <= 31) | (~df['SubRegion'].isin(exclude_subregions))]

# Remove rows with 'mean_nitrate' as NaN if any
df = df.dropna(subset=['mean_nitrate'])

# %%
# Increase the global font size
plt.rcParams.update({'font.size': 13})

# Creating a new column for grouping
df['Resistivity_Group'] = df['Resistivity_lyrs_9_rad_2_miles'].apply(lambda x: 'Above 50' if x > 20 else 'Below 50')

plt.figure(figsize=(7, 8))
ax = sns.boxplot(data=df, x='Resistivity_Group', y='mean_nitrate')
ax.set_yscale('log')
plt.title('Boxplot of Mean Nitrate for Resistivity Above and Below 50')
plt.xlabel('Resistivity Group')
plt.ylabel('Mean Nitrate (log scale)')

# Annotate with number of observations
n_obs = df['Resistivity_Group'].value_counts().values
for i, v in enumerate(n_obs):
    ax.text(i, 0.95, f'n={v}', color='black', ha='center', transform=ax.get_xaxis_transform())

# Set y limit to be 1.3 times the max value
ax.set_ylim(top=np.max(df['mean_nitrate']) * 1.3)

# Remove rows with 'mean_nitrate' as NaN if any
df = df.dropna(subset=['mean_nitrate'])

# Perform Mann-Whitney U Test only if there's more than one unique value in both groups and more than one sample per group
above_50 = df.loc[df['Resistivity_Group'] == 'Above 50', 'mean_nitrate']
below_50 = df.loc[df['Resistivity_Group'] == 'Below 50', 'mean_nitrate']

if len(above_50.unique()) > 1 and len(below_50.unique()) > 1 and len(above_50) > 1 and len(below_50) > 1:
    mwu_result = mannwhitneyu(above_50, below_50)

    # If p-value < 0.05, we reject the null hypothesis and conclude that there's a statistical difference
    if mwu_result.pvalue < 0.05:
        print(f'The p-value is {mwu_result.pvalue}. There is a statistically significant difference in mean_nitrate between the groups.')
    else:
        print(f'The p-value is {mwu_result.pvalue}. There is no statistically significant difference in mean_nitrate between the groups.')
else:
    print('Mann-Whitney U Test cannot be performed due to identical values or insufficient data in one or both groups.')
    
plt.show()

# %%
