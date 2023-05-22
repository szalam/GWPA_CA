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
plt.rcParams.update({'font.size': 16})

# creating a new column for grouping
df['Grouped_GWPAType'] = df['GWPAType'].apply(lambda x: x if x == 'Leaching' else 'Outside Leaching GWPA')

plt.figure(figsize=(8, 6))
ax = sns.boxplot(data=df, x='Grouped_GWPAType', y='mean_nitrate')
ax.set_yscale('log')

ax.set_ylim(top=np.max(df['mean_nitrate']) * 2)

plt.title('Mean Nitrate inside/outside leaching GWPA')
plt.xlabel(' ')
plt.ylabel('Mean Nitrate-N (mg/l)')

# Annotate with number of observations
n_obs = df['Grouped_GWPAType'].value_counts().values
for i, v in enumerate(n_obs):
    ax.text(i, .95, f'n={v}', color='black', ha='center', transform=ax.get_xaxis_transform())

# Perform Mann-Whitney U Test
leaching = df.loc[df['Grouped_GWPAType'] == 'Leaching', 'mean_nitrate']
outside = df.loc[df['Grouped_GWPAType'] == 'Outside Leaching GWPA', 'mean_nitrate']
mwu_result = mannwhitneyu(leaching, outside)

# If p-value < 0.05, we reject the null hypothesis and conclude that there's a statistical difference
if mwu_result.pvalue < 0.05:
    print(f'The p-value is {mwu_result.pvalue}. There is a statistically significant difference in mean_nitrate between Leaching and Outside GWPA.')
else:
    print(f'The p-value is {mwu_result.pvalue}. There is no statistically significant difference in mean_nitrate between Leaching and Outside GWPA.')
    
plt.show()

# %%
# Remove rows with 'mean_nitrate' as NaN if any
df = df.dropna(subset=['mean_nitrate'])

df_leach = df[df.GWPAType == 'Leaching']
# %%
df_leach.Resistivity_lyrs_9_rad_2_miles.describe()
#%%
df.Resistivity_lyrs_9_rad_2_miles.describe()
# %%
# =======================================================
# Plot resistivity for all wells nd for those inside GWPA
# =======================================================
# Create a new column for grouping
df['Grouped_GWPAType'] = 'All wells'
df_leach['Grouped_GWPAType'] = 'Inside Leaching GWPA'

# Concatenate the two dataframes
combined_df = pd.concat([df, df_leach])

# Increase the global font size
plt.rcParams.update({'font.size': 16})

plt.figure(figsize=(8, 6))
ax = sns.boxplot(data=combined_df, x='Grouped_GWPAType', y='Resistivity_lyrs_9_rad_2_miles')


# Set any desired plot configurations
plt.title('Depth Average Resistivity Distribution')
plt.xlabel(' ')
plt.ylabel('Depth Average Resistivity (ohm-m)')

ax.set_yscale('log')
ax.set_ylim(top=np.max(df['mean_nitrate']) * 2)
# Annotate with number of observations
n_obs = combined_df['Grouped_GWPAType'].value_counts().values
for i, v in enumerate(n_obs):
    ax.text(i, .95, f'n={v}', color='black', ha='center', transform=ax.get_xaxis_transform())

# Perform Mann-Whitney U Test
leaching = df_leach['Resistivity_lyrs_9_rad_2_miles']
outside = df['Resistivity_lyrs_9_rad_2_miles']
mwu_result = mannwhitneyu(leaching, outside)

# If p-value < 0.05, we reject the null hypothesis and conclude that there's a statistical difference
if mwu_result.pvalue < 0.05:
    print(f'The p-value is {mwu_result.pvalue}. There is a statistically significant difference in Resistivity between Leaching and Outside GWPA.')
else:
    print(f'The p-value is {mwu_result.pvalue}. There is no statistically significant difference in Resistivity between Leaching and Outside GWPA.')

# Show the plot
plt.show()
# %%
