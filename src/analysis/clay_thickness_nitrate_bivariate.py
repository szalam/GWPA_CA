#================================================================================
# This script analyzies the relationship between animal population and nitrate
# for wells when they have almost similar conductivity
#================================================================================
#%%
# Import required modules
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib as mpl
from scipy import stats
import statsmodels.api as sm

# Local modules
sys.path.insert(0,'src')
sys.path.insert(0, 'src/data')
import config
import ppfun as dp
import get_infodata as gi

#%%
# Functions
def load_data(version):
    """Load data based on version"""
    filename = "Dataset_processed_GAMAlatest.csv" if version == 2 else "Dataset_processed.csv"
    df = pd.read_csv(config.data_processed / filename)
    return df

def filter_data(df, well_type,all_dom_flag):
    """Filter and modify data"""
    exclude_subregions = [14, 15, 10, 19, 18, 9, 6]
    if all_dom_flag == 2:
        df = df[df.well_type ==  well_type] 
    df = df[(df[f'thickness_abovCond_{round(.1*100)}_lyrs_9_rad_2miles'] <= 31) | (~df['SubRegion'].isin(exclude_subregions))]
    return df

# Constants
# plt_cond = 0.1      # Threshold condoctivity above which thickness calculated
lyrs = 9
rad_buffer = 2
gama_old_new = 2
all_dom_flag = 2 # 1: All, 2: Domestic
if all_dom_flag == 2:
    well_type_select = {1: 'Domestic', 2: 'DOMESTIC'}.get(gama_old_new)
else:
    well_type_select = 'All'
cond_type_used = 'Resistivity_lyrs_9_rad_2_miles'
aem_type = 'Conductivity' if 'Conductivity' in cond_type_used else 'Resistivity'

# Load and process data
df_main = load_data(gama_old_new)
df = df_main[df_main.well_data_source == 'GAMA'].copy()
# separate wells inside cv
well_cv = pd.read_csv(config.data_processed / 'wells_inside_CV_GAMAlatest.csv',index_col=False)
# Assuming df is your dataframe with all wells
df_cv = df[df['well_id'].isin(well_cv['well_id'])]

df = filter_data(df_cv, well_type_select,all_dom_flag)

layer_depths = gi.dwr_aem_depths()


#%%

def plot_heatmap(df, plt_conds, interval, stat,max_thickness,rad_buffer,lyrs,all_dom_flag):
    thickness_intervals = range(0, max_thickness, interval)

    # create a dictionary of column names for each plt_cond
    column_dict = {plt_cond: f'thickness_abovCond_{round(plt_cond*100)}_lyrs_{lyrs}_rad_{rad_buffer}miles' 
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
    fig, ax = plt.subplots(figsize=(10, 8))
    # im = ax.imshow(nitrate_array, cmap='coolwarm')
    # im = ax.imshow(nitrate_array, cmap='RdBu_r')
    im = ax.imshow(nitrate_array, cmap='viridis', vmax = 5, vmin = 0)
    # set x and y ticks and labels
    ax.set_xticks(np.arange(len(thickness_intervals)))
    ax.set_xticklabels(thickness_intervals)
    ax.set_xlabel(r'Thickness with resistivity >= $\rho_{\mathrm{th}}$ ',fontsize =22)
    ax.set_yticks(np.arange(len(plt_conds)))
    ax.set_yticklabels([round(1/plt_cond,1) for plt_cond in plt_conds])
    ax.set_ylabel(r'Resistivity threshold ($\rho_{\mathrm{th}}$)',fontsize =22)
    plt.tick_params(axis='both', which='major', labelsize=19)

    
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
        if all_dom_flag == 1:
            plt.title('All wells', fontsize=24)
        if all_dom_flag == 2:
            plt.title('Domestic wells', fontsize=24)
        # add colorbar
        cbar = ax.figure.colorbar(im, ax=ax,shrink=0.5)
        cbar.ax.tick_params(labelsize=14)
        cbar.ax.set_ylabel(f'{stat.capitalize()} Nitrate-N', rotation=-90, va='bottom', fontsize = 22)
    # show plot
    plt.show()

#%%
plot_heatmap(df, plt_conds=[0.05, 0.06, 0.07, 0.08, 0.1], interval=3, stat='median',max_thickness = 30, rad_buffer = 2,lyrs = 9,all_dom_flag=all_dom_flag)
# plot_heatmap(df, plt_conds=[0.05, 0.06, 0.07, 0.08, 0.1], interval=3, stat='75th percentile',max_thickness = 30,rad_buffer = 2,lyrs = 9)
# plot_heatmap(df, plt_conds=[0.05, 0.08, 0.1], interval=3, stat='std',max_thickness = 30, rad_buffer = 2)
# plot_heatmap(df, plt_conds=[0.05, 0.06, 0.07, 0.08, 0.1], interval=3, stat='max',max_thickness = 30,rad_buffer = 2,lyrs = 9)
# plot_heatmap(df, plt_conds=[0.05, 0.06, 0.07, 0.08, 0.1], interval=3, stat='count',max_thickness = 30, rad_buffer = 1, lyrs = 6)


# %%

# For the following heatmap, taking values of 0 thickness for a lower resistivity to estimate the median for next higher resistivity
def plot_heatmap2(df, plt_conds, interval, stat,max_thickness,rad_buffer,lyrs):
    thickness_intervals = range(0, max_thickness, interval)

    # create a dictionary of column names for each plt_cond
    column_dict = {plt_cond: f'thickness_abovCond_{round(plt_cond*100)}_lyrs_{lyrs}_rad_{rad_buffer}miles' 
                   for plt_cond in plt_conds}

    # create a 2D array to hold the nitrate values for each plt_cond and thickness interval
    nitrate_array = np.zeros((len(plt_conds), len(thickness_intervals)))

    # create a 2D array to hold the count of measurements for each plt_cond and thickness interval
    count_array = np.zeros((len(plt_conds), len(thickness_intervals)))

    # iterate over each plt_cond and thickness interval and fill in the nitrate_array
    for i, plt_cond in enumerate(plt_conds):
        column_name = column_dict[plt_cond]
        if i!=(len(plt_conds)-1):
            column_name_next = column_dict[plt_conds[i+1]]
        for j, thickness in enumerate(thickness_intervals):
            if i!=(len(plt_conds)-1):
                mask = (df[column_name_next] < 0.00001) & (df[column_name] >= thickness) & (df[column_name] < thickness + interval)
            else:
                mask = (df[column_name] >= thickness) & (df[column_name] < thickness + interval)
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
    fig, ax = plt.subplots(figsize=(6, 5))
    # im = ax.imshow(nitrate_array, cmap='coolwarm')
    im = ax.imshow(nitrate_array, cmap='viridis')
    # set x and y ticks and labels
    ax.set_xticks(np.arange(len(thickness_intervals)))
    ax.set_xticklabels(thickness_intervals)
    ax.set_xlabel(r'Thickness with resistivity >= $\rho_{\mathrm{th}}$ ')
    ax.set_yticks(np.arange(len(plt_conds)))
    ax.set_yticklabels([round(1/plt_cond,1) for plt_cond in plt_conds])
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
        cbar = ax.figure.colorbar(im, ax=ax,shrink=0.6)
        cbar.ax.tick_params(labelsize=10)  
        cbar.ax.set_ylabel(f'{stat.capitalize()} Nitrate-N', rotation=-90, va='bottom')
    # show plot
    plt.show()
# %%
plot_heatmap2(df, plt_conds=[0.05, 0.06, 0.07, 0.08, 0.1], interval=3, stat='median',max_thickness = 30, rad_buffer = 2,lyrs = 9)

# %%

# running code considering all well buffer when calculating median, mean for any interval
def plot_heatmap3(df, plt_conds, interval, stat, max_thickness, rad_buffers=[2, 3, 4], lyrs=[4, 6, 9]):
    thickness_intervals = range(0, max_thickness, interval)
    num_plt_conds = len(plt_conds)

    # create a 3D array to hold the nitrate values for each plt_cond, thickness interval, and (rad_buffer, lyrs) pair
    nitrate_array = np.zeros((num_plt_conds, len(thickness_intervals), len(rad_buffers) * len(lyrs)))

    # iterate over each plt_cond, thickness interval, and (rad_buffer, lyrs) pair and fill in the nitrate_array
    for i, plt_cond in enumerate(plt_conds):
        for j, thickness in enumerate(thickness_intervals):
            for k, rad_buffer in enumerate(rad_buffers):
                for l, lyr in enumerate(lyrs):
                    column_name = f'thickness_abovCond_{round(plt_cond*100)}_lyrs_{lyr}_rad_{rad_buffer}miles'
                    mask = (df[column_name] >= thickness) & (df[column_name] < thickness+interval)
                    nitrate_values = df.loc[mask, 'mean_nitrate']

                    if stat == 'median':
                        nitrate_array[i, j, k*len(lyrs)+l] = nitrate_values.median()
                    if stat == 'mean':
                        nitrate_array[i, j, k*len(lyrs)+l] = nitrate_values.mean()
                    if stat == 'std':
                        nitrate_array[i, j, k*len(lyrs)+l] = nitrate_values.std()

    # calculate the median nitrate value across all (rad_buffer, lyrs) pairs for each plt_cond and thickness interval
    median_nitrate_array = np.median(nitrate_array, axis=2)

    # create a heatmap using the median nitrate array
    fig, ax = plt.subplots()
    im = ax.imshow(median_nitrate_array, cmap='RdBu_r')

    # set x and y ticks and labels
    ax.set_xticks(np.arange(len(thickness_intervals)))
    ax.set_xticklabels(thickness_intervals)
    ax.set_xlabel(r'Thickness with resistivity >= $\rho_{\mathrm{th}}$ ')
    ax.set_yticks(np.arange(num_plt_conds))
    ax.set_yticklabels([round(1/plt_cond,1) for plt_cond in plt_conds])
    ax.set_ylabel(r'Resistivity threshold ($\rho_{\mathrm{th}}$)')

    # set title
    ax.set_title(f'{stat.capitalize()} Nitrate-N Median')

    # add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Nitrate-N', rotation=-90, va='bottom')

    # show plot
    plt.show()

# plot_heatmap3(df, plt_conds=[0.05, 0.06, 0.07, 0.08, 0.1], interval=3, stat='median',max_thickness = 30, rad_buffers=[0.5,1,2,3,4,5], lyrs=[4])
plot_heatmap3(df, plt_conds=[0.05, 0.06, 0.07, 0.08, 0.1], interval=3, stat='std',max_thickness = 30, rad_buffers=[0.5,1,2,3,4,5], lyrs=[9])

# %%
