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
gama_old_new = 2
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
df = filter_data(df, well_type_select,all_dom_flag)

#%%
# separate wells inside cv
well_cv = pd.read_csv(config.data_processed / 'wells_inside_CV_GAMAlatest.csv',index_col=False)
# Assuming df is your dataframe with all wells
df_cv = df[df['well_id'].isin(well_cv['well_id'])]

#%%
df = df_cv.copy()
df = df.drop(['well_id', 'well_data_source','start_date', 'end_date'], axis=1)

# Remove rows with 'mean_nitrate' as NaN if any
df = df.dropna(subset=['mean_nitrate'])

# %%

# creating a new column for grouping
df['Grouped_GWPAType'] = df['GWPAType'].apply(lambda x: x if x == 'Leaching' else 'Outside Leaching GWPA')


# %%
# Remove rows with 'mean_nitrate' as NaN if any
df = df.dropna(subset=['mean_nitrate'])
df = df.dropna(subset=['Resistivity_lyrs_9_rad_2_miles'])

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

#%%
# Increase the global font size
plt.rcParams.update({'font.size': 19})

plt.figure(figsize=(7, 7))
ax = sns.boxplot(data=combined_df, x='Grouped_GWPAType', y='Resistivity_lyrs_9_rad_2_miles')


# Set any desired plot configurations
# plt.title('Depth Average Resistivity Distribution')
plt.xlabel(' ')
plt.ylabel('Depth Average Resistivity (ohm-m)', fontsize = 22)

ax.set_yscale('log')
ax.set_ylim(top=np.max(df['mean_nitrate']) * 2)
# Annotate with number of observations
n_obs = combined_df['Grouped_GWPAType'].value_counts().values
for i, v in enumerate(n_obs):
    ax.text(i, .95, f'n={v}', color='black', ha='center', transform=ax.get_xaxis_transform())

# Show the plot
plt.show()

# Perform Mann-Whitney U Test
leaching = df_leach['Resistivity_lyrs_9_rad_2_miles']
outside = df['Resistivity_lyrs_9_rad_2_miles']

#%%
# Checking NaN values

# Checking 'df' dataframe
if df['Resistivity_lyrs_9_rad_2_miles'].isna().any():
    print("df dataframe has NaN values in 'Resistivity_lyrs_9_rad_2_miles' column")
if df['Resistivity_lyrs_9_rad_2_miles'].nunique() <= 1:
    print("'df' dataframe has 1 or less unique values in 'Resistivity_lyrs_9_rad_2_miles' column")

# Checking 'df_leach' dataframe
if df_leach['Resistivity_lyrs_9_rad_2_miles'].isna().any():
    print("df_leach dataframe has NaN values in 'Resistivity_lyrs_9_rad_2_miles' column")
if df_leach['Resistivity_lyrs_9_rad_2_miles'].nunique() <= 1:
    print("'df_leach' dataframe has 1 or less unique values in 'Resistivity_lyrs_9_rad_2_miles' column")

mwu_result = mannwhitneyu(leaching, outside)

# If p-value < 0.05, we reject the null hypothesis and conclude that there's a statistical difference
if mwu_result.pvalue < 0.05:
    print(f'The p-value is {mwu_result.pvalue}. There is a statistically significant difference in Resistivity between Leaching and Outside GWPA.')
else:
    print(f'The p-value is {mwu_result.pvalue}. There is no statistically significant difference in Resistivity between Leaching and Outside GWPA.')


# %%
