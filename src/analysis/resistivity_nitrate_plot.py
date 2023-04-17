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

df_main = pd.read_csv(config.data_processed / "Dataset_processed.csv")
#%%
df = df_main.copy()
lyrs = 9
rad_well = 2
cond_type_used = f'Resistivity_lyrs_{lyrs}' #'Conductivity_lyrs_9' # 'Conductivity_lyrs_1'
# cond_type_used = 'Conductivity_lyrs_9'
if cond_type_used == 'Conductivity_lyrs_9' or cond_type_used == 'Conductivity_lyrs_1':
    aem_type = 'Conductivity'
else:
    aem_type = 'Resistivity'

# Read dataset
df = df[df.well_data_source == 'GAMA']
# df = df[df.measurement_count > 4]
# df = df[df.city_inside_outside == 'outside_city']
# df = df[df.city_inside_outside == 'inside_city']
# df = df[[f'{cond_type_used}','mean_nitrate']]
# df = df[df['SubRegion'] != 14]
# df = df[df['SubRegion']==6]
# df = df[df[f'thickness_abovCond_{round(.1*100)}'] == 0 ]
well_type_select = 'Domestic' # 'Water Supply, Other', 'Municipal', 'Domestic'
df = df[df.well_type ==  well_type_select] 
# df = df[df.All_crop_2015 >= 2.030499e+08]

# Remove high salinity regions
exclude_subregions = [14, 15, 10, 19,18, 9,6]
# filter aemres to keep only the rows where Resistivity is >= 10 and SubRegion is not in the exclude_subregions list
df = df[(df[f'thickness_abovCond_{round(.1*100)}_lyrs_9_rad_{rad_well}miles'] <= 31) | (~df['SubRegion'].isin(exclude_subregions))]
# df = df[df.mean_nitrate>10]
# df = df.dropna()



# df = df[['well_id','Conductivity','mean_nitrate','area_wt_sagbi', 'Cafo_Population_5miles','Average_ag_area','change_per_year','total_ag_2010','APPROXIMATE LATITUDE', 'APPROXIMATE LONGITUDE','city_inside_outside']]
# df = df.head(1000)
#%%
plt.scatter(df[cond_type_used], df.mean_nitrate, s = 1.5, c = 'red')
plt.ylim(0 ,100)
# plt.xlim(0 ,70)
# plt.xlim(0 ,100)
# plt.xscale('log')


# Set x-axis to log scale
# plt.xscale('log')
# plt.yscale('log')
plt.xlabel(f'{aem_type}')
plt.ylabel('Nitrate')

#%%
# Generate cumulative distribution
plt.figure()
plt.hist(df[cond_type_used], bins=50, cumulative=True, density=True, histtype='step', color='blue')

# Set x-axis to log scale
plt.xscale('log')

# Set y-axis limit to 0 to 1
plt.ylim(0, 1)

# Show plot
plt.show()

# %%
if aem_type == 'Resistivity':
    df = df[df[f'{cond_type_used}']<80]

# Bin Conductivity into intervals of .05
if aem_type == 'Conductivity':
    df = df[df[f'{cond_type_used}']<1]
    df['Conductivity_binned'] = pd.cut(df[f'{cond_type_used}'], 
                                       np.arange(0, df[f'{cond_type_used}'].max()+.001,.01))
if aem_type == 'Resistivity':
    df['Conductivity_binned'] = pd.cut(df[f'{cond_type_used}'], 
                                       np.arange(0, df[f'{cond_type_used}'].max()+1,5))
# Increase the font size
# mpl.rcParams.update({'font.size': 14})

# Create a box and whisker plot using Seaborn
sns.boxplot(x='Conductivity_binned', y='mean_nitrate', data=df, width=0.5, color = 'orange')

# Add x and y labels
plt.xlabel(f'Depth Average {aem_type} (\u2126-m)', fontsize = 13)
plt.ylabel('Nitrate-N [mg/l]', fontsize =13)
plt.tick_params(axis='both', which='major', labelsize=10)

# Set y-axis to log scale
plt.yscale('log')
# Set y-axis limit to 0 to 100
plt.ylim(0, 1500)

# Rotate x tick labels for better readability
plt.xticks(rotation=90)

# Show plot
plt.show()

#%%
if aem_type == 'Conductivity':
    df[f"{cond_type_used}"].plot.hist(rwidth=0.9, color="orange", edgecolor='black', 
                                      bins=np.arange(df[f'{cond_type_used}'].min(), df[f'{cond_type_used}'].max() + .1, 0.02))
if aem_type == 'Resistivity':
    # df = df[df[f"{cond_type_used}"]<=100]
    df[f"{cond_type_used}"].plot.hist(rwidth=0.9, color="orange", edgecolor='black',  
                                      bins=np.arange(df[f'{cond_type_used}'].min(), df[f'{cond_type_used}'].max() + 1, 5))
 

plt.xlabel(f'{aem_type}')
plt.ylabel('Counts')
plt.title(f'{aem_type} Histogram')
plt.show()

#%%
# Test if the correlation coefficient is significant 

# Remove rows with missing data
cleaned_df = df[[f'{cond_type_used}', 'mean_nitrate']].dropna()

spearman_correlation_coefficient, spearman_p_value = stats.spearmanr(cleaned_df[f'{cond_type_used}'], cleaned_df['mean_nitrate'])

print(f"Spearman Correlation Coefficient: {spearman_correlation_coefficient}")
print(f"Spearman P-value: {spearman_p_value}")

alpha = 0.05
if spearman_p_value < alpha:
    print("The Spearman correlation is significant.")
else:
    print("The Spearman correlation is not significant.")


#%%
#=====================
# Test for statistical differences in mean

df_tmp = df.copy()
results = []

# Loop over conductivity_cutoff values from 5 to 40 at 5 intervals
for conductivity_cutoff in range(10, 50, 5):
    # Split the data into two groups based on conductivity
    low_cond = df_tmp[df_tmp[f'{cond_type_used}'] < conductivity_cutoff]['mean_nitrate']
    high_cond = df_tmp[df_tmp[f'{cond_type_used}'] >= conductivity_cutoff]['mean_nitrate']

    # Remove any non-numeric data and missing values
    low_cond = low_cond.apply(pd.to_numeric, errors='coerce').dropna()
    high_cond = high_cond.apply(pd.to_numeric, errors='coerce').dropna()

    # Perform a two-sample t-test
    t_stat, p_value = stats.ttest_ind(low_cond, high_cond, equal_var=False)

    # Interpret the results
    if p_value < 0.05:
        significance = "significant"
    else:
        significance = "not significant"

    results.append([conductivity_cutoff, p_value, significance])

# Create a DataFrame with the results and display it
results_df = pd.DataFrame(results, columns=['cutoff', 'p_value', 'significance'])
print(results_df)

#%%
# Test for statistical differences in median

df_tmp = df.copy()
results = []

# Loop over conductivity_cutoff values from 5 to 40 at 5 intervals
for conductivity_cutoff in range(5, 45, 5):
    # Split the data into two groups based on conductivity
    low_cond = df_tmp[df_tmp[f'{cond_type_used}'] < conductivity_cutoff]['mean_nitrate']
    high_cond = df_tmp[df_tmp[f'{cond_type_used}'] >= conductivity_cutoff]['mean_nitrate']

    # Remove any non-numeric data and missing values
    low_cond = low_cond.apply(pd.to_numeric, errors='coerce').dropna()
    high_cond = high_cond.apply(pd.to_numeric, errors='coerce').dropna()

    # Perform a Mann-Whitney U test
    u_stat, p_value = stats.mannwhitneyu(low_cond, high_cond, alternative='two-sided')

    # Interpret the results
    if p_value < 0.05:
        significance = "significant"
    else:
        significance = "not significant"

    results.append([conductivity_cutoff, p_value, significance])

# Create a DataFrame with the results and display it
results_df = pd.DataFrame(results, columns=['cutoff', 'p_value', 'significance'])
print(results_df)


# %%
# Loess plot
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Create scatter plot
plt.scatter(df[cond_type_used], df.mean_nitrate, s=1.5, c='red')
plt.ylim(0, 100)
plt.xlabel(f'{aem_type}')
plt.ylabel('Nitrate')

# Fit loess model and generate predictions
lowess = sm.nonparametric.lowess(df.mean_nitrate, df[cond_type_used], frac=0.3)
x_pred = lowess[:, 0]
y_pred = lowess[:, 1]

# Add regression line to plot
plt.plot(x_pred, y_pred, color='blue')

# Set x-axis to log scale
# plt.xscale('log')

# Show plot
plt.show()



# %%
# Plot truncated gaussian distribution
import numpy as np
import scipy.stats as stats


# Truncate the conductivity data to the range [0, cond_range]
cond_range = .8
truncated_conductivity = stats.truncnorm.rvs(
    (0 - df['Conductivity'].mean()) / df['Conductivity'].std(), 
    (cond_range - df['Conductivity'].mean()) / df['Conductivity'].std(), 
    loc=df['Conductivity'].mean(), 
    scale=df['Conductivity'].std(), 
    size=len(df)
)

# Truncate the nitrate data to the range [0, n_range]
n_range = 1000
truncated_nitrate = stats.truncnorm.rvs(
    (0 - df['mean_nitrate'].mean()) / df['mean_nitrate'].std(), 
    (n_range - df['mean_nitrate'].mean()) / df['mean_nitrate'].std(), 
    loc=df['mean_nitrate'].mean(), 
    scale=df['mean_nitrate'].std(), 
    size=len(df)
)

# Plot the truncated data
plt.scatter(truncated_conductivity, truncated_nitrate, s = 1.5, c = 'red')
plt.ylim(0 ,n_range)
plt.xlabel('Conductivity')
plt.ylabel('Nitrate')

# Show plot
plt.show()

#%%
# How many nan are there in the conductivity column?
nan_count = np.sum(df['Conductivity'].isna())
print("Number of NaN values in Conductivity column: ", nan_count)

# %%
#====================================
import statsmodels.api as sm
import pandas as pd
import numpy as np

# Define conductivity ranges
conductivity_ranges = np.arange(0, 1, 0.1)

df2 = df[['Conductivity','mean_nitrate']]
df2 = df2.dropna()

# Create a new column with the conductivity range
df2['conductivity_range'] = pd.cut(df2['Conductivity'], bins=conductivity_ranges)
df2['conductivity_range'] = df2['conductivity_range'].astype(str)
df2['conductivity_range'] = df2['conductivity_range'].str.extract(r'\((.*?),', expand=False).astype(float)

# Fit a hierarchical linear model
model = sm.MixedLM.from_formula("mean_nitrate ~ Conductivity", groups=df2['conductivity_range'], data=df2)
result = model.fit()

# Check the results
print(result.summary())

# Output from the above
#             Mixed Linear Model Regression Results
# =============================================================
# Model:             MixedLM  Dependent Variable:  mean_nitrate
# No. Observations:  6170     Method:              REML        
# No. Groups:        8        Scale:               50855.3413  
# Min. group size:   2        Log-Likelihood:      -42185.5642 
# Max. group size:   5755     Converged:           Yes         
# Mean group size:   771.2                                     
# -------------------------------------------------------------
#                Coef.   Std.Err.   z    P>|z|  [0.025   0.975]
# -------------------------------------------------------------
# Intercept      -38.787   83.739 -0.463 0.643 -202.912 125.337
# Conductivity   332.025  175.969  1.887 0.059  -12.868 676.918
# Group Var    12050.556   66.504                              
# =============================================================



# This result indicates that there is a relationship between conductivity 
# and nitrate, with conductivity having a positive effect on nitrate, although 
# this effect is not statistically significant (p-value = 0.059, which is greater 
# than the commonly used threshold of 0.05). The coefficients section gives the
# estimated mean effect of conductivity on nitrate, with a standard error. 
# The "z" and "P>|z|" values give a test of statistical significance, with a small
# p-value indicating strong evidence against the null hypothesis (in this case, 
# that the true effect of conductivity on nitrate is zero). The "[0.025 0.975]" 
# column gives 95% confidence intervals for the estimated mean effect, which can 
# be used to determine the range of values that is likely to contain the true 
# effect with a high degree of confidence.

# %%
# Nitrate difference below vs above threshold
from scipy.stats import ttest_ind

for res_threshold in [10, 15, 20, 25, 30, 35, 40]:
    df_up = df[df[cond_type_used] > res_threshold]
    df_down = df[df[cond_type_used] <= res_threshold]
    mean_nitrate_up = df_up.mean_nitrate.mean()
    mean_nitrate_down = df_down.mean_nitrate.mean()
    median_nitrate_up = df_up.mean_nitrate.median()
    median_nitrate_down = df_down.mean_nitrate.median()
    t, p = ttest_ind(df_up.mean_nitrate, df_down.mean_nitrate, equal_var=False)
    if p < 0.05:
        print(f'Resistivity: {res_threshold}; up: {median_nitrate_up:.2f}, down: {median_nitrate_down:.2f}; up-down: {median_nitrate_up - median_nitrate_down:.2f}; p-value: {p:.3f} (stat significant)')
    else:
        print(f'Resistivity: {res_threshold}; up: {median_nitrate_up:.2f}, down: {median_nitrate_down:.2f}; up-down: {median_nitrate_up - median_nitrate_down:.2f}; p-value: {p:.3f} (not stat significant)')
#%%
# Nitrate difference +- 10 of the threshold
for res_threshold in [20, 30, 40,50]:
    df_up = df[(df[cond_type_used] > res_threshold) & (df[cond_type_used] <= res_threshold + 10)]
    df_down = df[(df[cond_type_used] >= res_threshold - 10) & (df[cond_type_used] <= res_threshold)]
    mean_nitrate_up = df_up.mean_nitrate.mean()
    mean_nitrate_down = df_down.mean_nitrate.mean()
    median_nitrate_up = df_up.mean_nitrate.median()
    median_nitrate_down = df_down.mean_nitrate.median()
    t, p = ttest_ind(df_up.mean_nitrate, df_down.mean_nitrate, equal_var=False)
    if p < 0.05:
        print(f'Resistivity: {res_threshold}; up: {median_nitrate_up:.2f}, down: {median_nitrate_down:.2f}; up-down: {median_nitrate_up - median_nitrate_down:.2f}; p-value: {p:.3f} (statistically significant)')
    else:
        print(f'Resistivity: {res_threshold}; up: {median_nitrate_up:.2f}, down: {median_nitrate_down:.2f}; up-down: {median_nitrate_up - median_nitrate_down:.2f}; p-value: {p:.3f} (not statistically significant)')




# %%
fig, ax = plt.subplots(1,1)
ax.hist(df_up.mean_nitrate,alpha = .5, density = True, bins = np.arange(40))
ax.hist(df_down.mean_nitrate, alpha = .5, density=True, bins = np.arange(40))
# %%
df_up.mean_nitrate.describe()
df_down.mean_nitrate.describe()
# %%
