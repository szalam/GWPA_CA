#================================================================================
# This script compares the resistivity with nitrate-n distribution around 
# wells (All wells vs domestic wells)
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
    """Filter data"""
    exclude_subregions = [14, 15, 10, 19, 18, 9, 6]
    if all_dom_flag == 2:
        df = df[df.well_type ==  well_type] 
    df = df[(df[f'thickness_abovCond_{round(.1*100)}_lyrs_9_rad_2miles'] <= 31) | (~df['SubRegion'].isin(exclude_subregions))]
    return df

# Constants
plt_cond = 0.1      # Threshold condoctivity above which thickness calculated
lyrs = 9
rad_buffer = 2
gama_old_new = 2
all_dom_flag = 1 # 1: All, 2: Domestic
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

# Functions
def plot_scatter(df, x_col, y_col, plt_cond):
    """Plot scatter plot"""
    plt.figure(figsize=(10, 8))
    plt.scatter(df[x_col], df[y_col], s=1.5, color='skyblue')
    plt.xlabel(f'Thickness of layers with resistivity <= {1/plt_cond}', fontsize=18)
    plt.ylabel('Nitrate-N', fontsize=18)
    plt.ylim(0, 40)
    plt.grid(True)
    plt.title('Nitrate concentration versus layer thickness', fontsize=20)
    plt.show()

def plot_lowess(df, x_col, y_col, plt_cond):
    """Plot scatter with LOWESS line"""
    plt.figure(figsize=(10, 8))
    x, y = df[x_col], df[y_col]
    lowess = sm.nonparametric.lowess(y, x, frac=.2) 
    plt.scatter(x, y, s=1.5, color='orange')
    plt.plot(lowess[:,0], lowess[:,1], c='r')
    plt.xlabel(f'Thickness of layers with resistivity <= {1/plt_cond}', fontsize=18)
    plt.ylabel('Nitrate-N', fontsize=18)
    plt.ylim(0, 40)
    plt.grid(True)
    plt.title('Nitrate concentration versus layer thickness with LOWESS line', fontsize=20)
    plt.show()

def plot_boxplot(df, x_col, y_col, plt_cond,all_dom_flag):
    """Plot boxplot"""
    plt.figure(figsize=(10, 8))
    df['Depth_range'] = pd.cut(df[x_col], bins=[-1, 0, 5, 10, 15, 25, 35], labels=['0', '(0,5]', '(5,10]', '(10,15]', '(15,25]', '(25,35]'])
    sns.boxplot(x='Depth_range', y=y_col, data=df, color='orange', linewidth=1,saturation=1.0)
    plt.xlabel(f'Thickness of layers with resistivity <= {1/plt_cond}', fontsize=24)
    plt.ylabel('Nitrate-N (mg/L)', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=19)
    plt.yscale('log')
    plt.ylim(0, 1500)
    plt.grid(True)
    if all_dom_flag == 1:
        plt.title('All wells', fontsize=26)
    if all_dom_flag == 2:
        plt.title('Domestic wells', fontsize=26)
    plt.show()
    plt.show()

# Variables
x_col = f'thickness_abovCond_{round(plt_cond*100)}_lyrs_{lyrs}_rad_{rad_buffer}miles'
y_col = 'mean_nitrate'

# Plots
plot_scatter(df, x_col, y_col, plt_cond)
plot_lowess(df, x_col, y_col, plt_cond)
plot_boxplot(df, x_col, y_col, plt_cond,all_dom_flag)


# %%
