#================================================================================
# This script analyzes the relationship between resistivity and nitrate-n 
# concentration around wells (All wells vs domestic wells)
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

# Local modules
sys.path.insert(0,'src')
sys.path.insert(0, 'src/data')
import config
import ppfun as dp

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

def plot_data(df, x_col, y_col):
    """Plot data"""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x_col, y=y_col, data=df, s=50, color='red')
    # plt.yscale('log')
    plt.ylim(0 ,100)
    plt.xlabel(f'{aem_type}', fontsize=14)
    plt.ylabel('Nitrate', fontsize=14)
    plt.title('Nitrate concentration versus AEM Type', fontsize=18)
    plt.show()

# Constants
gama_old_new = 2
all_dom_flag = 2 # 1: All, 2: Domestic
if all_dom_flag == 2:
    well_type_select = {1: 'Domestic', 2: 'DOMESTIC'}.get(gama_old_new) 
if all_dom_flag == 1:
    well_type_select = 'All'
cond_type_used = 'Resistivity_lyrs_9_rad_2_miles'
aem_type = 'Conductivity' if 'Conductivity' in cond_type_used else 'Resistivity'

# Load and process data
df_main = load_data(gama_old_new)
df = df_main[df_main.well_data_source == 'GAMA'].copy()

#%%
# separate wells inside cv
well_cv = pd.read_csv(config.data_processed / 'wells_inside_CV_GAMAlatest.csv',index_col=False)
# Assuming df is your dataframe with all wells
df_cv = df[df['well_id'].isin(well_cv['well_id'])]

df = filter_data(df_cv, well_type_select,all_dom_flag)

# Plot data
plot_data(df, cond_type_used, 'mean_nitrate')

# %%
# Functions
def process_data(df, aem_type, cond_type_used):
    """Pre-process data"""
    if aem_type == 'Resistivity':
        df = df[df[cond_type_used] < 70]
        bin_width = 5
    else:
        df = df[df[cond_type_used] < 1]
        bin_width = .01
    df['Conductivity_binned'] = pd.cut(df[cond_type_used], np.arange(0, df[cond_type_used].max()+bin_width, bin_width))
    return df

def plot_boxplot(df, x_col, y_col, aem_type,all_dom_flag):
    """Plot boxplot"""
    plt.figure(figsize=(10, 8))
    sns.boxplot(x=x_col, y=y_col, data=df, width=0.5, color = 'orange')
    plt.xlabel(f'Depth Average {aem_type} (\u2126-m) for ~32m', fontsize = 24)
    plt.ylabel('Nitrate-N [mg/l]', fontsize = 24)
    plt.yscale('log')
    plt.ylim(0, 1500)
    plt.tick_params(axis='both', which='major', labelsize=19)
    plt.xticks(rotation=90)
    if all_dom_flag == 1:
        plt.title('(a) All wells', fontsize=26)
    if all_dom_flag == 2:
        plt.title('(b) Domestic wells', fontsize=26)
    plt.grid(axis='y')
    plt.show()

def plot_histogram(df, cond_type_used, aem_type):
    """Plot histogram"""
    bin_width = 5 if aem_type == 'Resistivity' else .02
    plt.figure(figsize=(10, 8))
    df[cond_type_used].plot.hist(rwidth=0.9, color="orange", edgecolor='black', bins=np.arange(df[cond_type_used].min(), df[cond_type_used].max() + bin_width, bin_width))
    plt.xlabel(f'{aem_type}', fontsize=18)
    plt.ylabel('Counts', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=19)
    plt.title(f'{aem_type} Histogram', fontsize=20)
    plt.grid(axis='y')
    plt.show()

# Process and plot data
df = process_data(df, aem_type, cond_type_used)
plot_boxplot(df, 'Conductivity_binned', 'mean_nitrate', aem_type, all_dom_flag = all_dom_flag)
plot_histogram(df, cond_type_used, aem_type)

# %%
