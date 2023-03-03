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
import get_infodata as gi
import geopandas as gpd

# Read dataset
df_main = pd.read_csv(config.data_processed / "Dataset_processed.csv")
#%%
df = df_main.copy()
plt_cond = 0.05      # Threshold condoctivity above which thickness calculated
lyrs = 9
#%%
df = df[df.well_data_source == 'GAMA']
# df = df[df['SubRegion'] != 14]
# df = df[df['SubRegion'] == 10]
# df = df[df.measurement_count > 4]
# df = df[df.city_inside_outside == 'outside_city']
well_type_select = 'Domestic' # 'Water Supply, Other', 'Municipal', 'Domestic'
df = df[df.well_type ==  well_type_select] 
# df = df[pd.isna(df.GWPAType)==True] # False for areas with GWPA

# Remove high salinity regions
exclude_subregions = [14, 15, 10, 19,18, 9,6]
# filter aemres to keep only the rows where Resistivity is >= 10 and SubRegion is not in the exclude_subregions list
df = df[(df[f'thickness_abovCond_{round(plt_cond*100)}_lyrs_9'] <= 31) | (~df['SubRegion'].isin(exclude_subregions))]
# df = df[(df[f'thickness_abovCond_{round(0.1*100)}_lyrs_{lyrs}'] == 0)]
# df = df[(df[f'thickness_abovCond_{round(0.08*100)}_lyrs_{lyrs}'] == 0)]
        
# condition = (df[f'thickness_abovCond_{round(plt_cond*100)}'] > 31) & (df['mean_nitrate'] > 0)
# df = df[condition==False]

df = df[df.mean_nitrate<100]
# df = df.dropna()

layer_depths = gi.dwr_aem_depths()

#%%
cond_threshold_fine_list = [0.01,0.02,0.03, 0.05,0.06,0.07, 0.08, 0.1, 0.13, 0.15, 0.18, 0.2, 0.25]

#%%
# assuming your dataframe is named 'df'
lyrs = 9
plt_conds = [0.05, 0.08, 0.1, 0.15]

# create a dictionary of column names for each plt_cond
column_dict = {plt_cond: f'thickness_abovCond_{round(plt_cond*100)}_lyrs_{lyrs}' 
               for plt_cond in plt_conds}

# iterate over each plt_cond and create a scatter plot
for plt_cond in plt_conds:
    # get the column name for this plt_cond
    column_name = column_dict[plt_cond]
    
    # calculate the median for each 5 intervals of thickness
    df['thickness_intervals'] = pd.cut(df[column_name], bins=range(0, 101, 5), right=False)
    median_nitrate = df.groupby('thickness_intervals')['mean_nitrate'].median()
    
    # create a scatter plot with point colors based on median_nitrate values
    plt.scatter(df[column_name], [plt_cond]*len(df), 
                c=df['thickness_intervals'].apply(lambda x: median_nitrate.loc[x]), cmap='coolwarm', s=1.5)
    
# add axis labels and a colorbar
plt.xlabel(f'Thickness of layers with resistivity <= {1/plt_cond}')
plt.ylabel('plt_cond')
plt.colorbar(label='Median Nitrate-N')

# show the plot
plt.show()

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_heatmap(df, plt_conds, interval, stat,max_thickness):
    lyrs = 9
    thickness_intervals = range(0, max_thickness, interval)

    # create a dictionary of column names for each plt_cond
    column_dict = {plt_cond: f'thickness_abovCond_{round(plt_cond*100)}_lyrs_{lyrs}' 
                   for plt_cond in plt_conds}

    # create a 2D array to hold the nitrate values for each plt_cond and thickness interval
    nitrate_array = np.zeros((len(plt_conds), len(thickness_intervals)))

    # create a 2D array to hold the count of measurements for each plt_cond and thickness interval
    count_array = np.zeros((len(plt_conds), len(thickness_intervals)))

    # iterate over each plt_cond and thickness interval and fill in the nitrate_array
    for i, plt_cond in enumerate(plt_conds):
        column_name = column_dict[plt_cond]
        for j, thickness in enumerate(thickness_intervals):
            mask = (df[column_name] >= thickness) & (df[column_name] < thickness+interval)
            nitrate_values = df.loc[mask, 'mean_nitrate']
            if stat == 'median':
                nitrate_array[i, j] = nitrate_values.median()
            elif stat == 'mean':
                nitrate_array[i, j] = nitrate_values.mean()
            elif stat == 'max':
                nitrate_array[i, j] = nitrate_values.max()
            elif stat == 'std':
                nitrate_array[i, j] = nitrate_values.std()
            elif stat == '25th percentile':
                nitrate_array[i, j] = np.percentile(nitrate_values, 25)
            elif stat == '75th percentile':
                nitrate_array[i, j] = np.percentile(nitrate_values, 75)
            elif stat == 'count':
                count_array[i, j] = len(nitrate_values)
    # create a heatmap using the nitrate_array
    fig, ax = plt.subplots()
    # im = ax.imshow(nitrate_array, cmap='coolwarm')
    im = ax.imshow(nitrate_array, cmap='viridis')
    # set x and y ticks and labels
    ax.set_xticks(np.arange(len(thickness_intervals)))
    ax.set_xticklabels(thickness_intervals)
    ax.set_xlabel(r'Thickness with resistivity >= $\rho_{\mathrm{th}}$ ')
    ax.set_yticks(np.arange(len(plt_conds)))
    ax.set_yticklabels([1/plt_cond for plt_cond in plt_conds])
    ax.set_ylabel(r'Resistivity threshold ($\rho_{\mathrm{th}}$)')

    
# set title
    if stat == 'count':
        ax.set_title(f'{stat.capitalize()} Nitrate-N and Counts')
        for i in range(len(plt_conds)):
            for j in range(len(thickness_intervals)):
                ax.text(j, i, int(count_array[i,j]), ha='center', va='center', color='w', fontsize=8)
        im = ax.imshow(count_array, cmap='viridis')  # updated line
        # add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Count', rotation=-90, va='bottom')  # updated line
    else:
        ax.set_title(f'{stat.capitalize()} Nitrate-N')
        # add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Nitrate-N', rotation=-90, va='bottom')
    # show plot
    plt.show()


plot_heatmap(df, plt_conds=[0.05, 0.08, 0.1], interval=3, stat='median',max_thickness = 30)
# plot_heatmap(df, plt_conds=[0.05, 0.08, 0.1], interval=3, stat='75th percentile',max_thickness = 30)
plot_heatmap(df, plt_conds=[0.05, 0.08, 0.1], interval=3, stat='std',max_thickness = 30)
plot_heatmap(df, plt_conds=[0.05, 0.08, 0.1], interval=3, stat='max',max_thickness = 30)
plot_heatmap(df, plt_conds=[0.05, 0.08, 0.1], interval=3, stat='count',max_thickness = 30)


# %%

# Plot for cases when thickness is zero for a given resistivity
def plot_heatmap2(df, plt_conds, interval, stat, max_thickness):
    lyrs = 9
    thickness_intervals = range(0, max_thickness, interval)

    # create a dictionary of column names for each plt_cond
    column_dict = {plt_cond: f'thickness_abovCond_{round(plt_cond*100)}_lyrs_{lyrs}' 
                   for plt_cond in plt_conds}

    # create a 2D array to hold the nitrate values for each plt_cond and thickness interval
    nitrate_array = np.zeros((len(plt_conds), len(thickness_intervals)))

    # create a 2D array to hold the count of measurements for each plt_cond and thickness interval
    count_array = np.zeros((len(plt_conds), len(thickness_intervals)))

    # iterate over each plt_cond and thickness interval and fill in the nitrate_array
    for i, plt_cond in enumerate(plt_conds):
        column_name = column_dict[plt_cond]
        if plt_cond == 0.05:
            thickness_cond = 0
        elif plt_cond == 0.08:
            thickness_cond = 0 if not any(df[column_dict[0.05]] > 0) else None
        else:
            thickness_cond = None
        for j, thickness in enumerate(thickness_intervals):
            mask = (df[column_name] >= thickness) & (df[column_name] < thickness+interval)
            if thickness_cond is not None:
                mask &= (df[column_dict[0.08]] == thickness_cond)
            nitrate_values = df.loc[mask, 'mean_nitrate']
            if stat == 'median':
                nitrate_array[i, j] = nitrate_values.median()
            elif stat == 'mean':
                nitrate_array[i, j] = nitrate_values.mean()
            elif stat == 'max':
                nitrate_array[i, j] = nitrate_values.max()
            elif stat == 'std':
                nitrate_array[i, j] = nitrate_values.std()
            elif stat == '25th percentile':
                nitrate_array[i, j] = np.percentile(nitrate_values, 25)
            elif stat == '75th percentile':
                nitrate_array[i, j] = np.percentile(nitrate_values, 75)
            elif stat == 'count':
                count_array[i, j] = len(nitrate_values)
    # create a heatmap using the nitrate_array
    fig, ax = plt.subplots()
    # im = ax.imshow(nitrate_array, cmap='coolwarm')
    im = ax.imshow(nitrate_array, cmap='viridis')
    # set x and y ticks and labels
    ax.set_xticks(np.arange(len(thickness_intervals)))
    ax.set_xticklabels(thickness_intervals)
    ax.set_xlabel(r'Thickness with resistivity >= $\rho_{\mathrm{th}}$ ')
    ax.set_yticks(np.arange(len(plt_conds)))
    ax.set_yticklabels([1/plt_cond for plt_cond in plt_conds])
    ax.set_ylabel(r'Resistivity threshold ($\rho_{\mathrm{th}}$)')

    # set title
    if stat == 'count':
        ax.set_title(f'{stat.capitalize()} Nitrate-N and Counts')
        for i in range(len(plt_conds)):
            for j in range(len(thickness_intervals)):
                ax.text(j, i, int(count_array[i,j]), ha='center', va='center', color='w', fontsize=8)
        im = ax.imshow(count_array, cmap='viridis')  # updated line
        # add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Count', rotation=-90, va='bottom')  # updated line
    else:
        ax.set_title(f'{stat.capitalize()} Nitrate-N')
        # add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Nitrate-N', rotation=-90, va='bottom')
    # show plot
    plt.show()


plot_heatmap2(df, plt_conds=[0.05, 0.08, 0.1], interval=3, stat='median',max_thickness = 30)
plot_heatmap2(df, plt_conds=[0.05, 0.08, 0.1], interval=3, stat='count',max_thickness = 30)

# %%
