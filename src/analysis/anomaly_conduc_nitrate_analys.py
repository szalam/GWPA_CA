#================================================================================
# This script analyzies the anomaly between conductivity vs nitrate
#================================================================================
#%%
import sys
sys.path.insert(0,'src')

import pandas as pd
import config
import geopandas as gpd
import matplotlib.pyplot as plt 
from tqdm import tqdm

from scipy.stats import ttest_ind
from visualize import vismod

# Read dataset
df = pd.read_csv(config.data_processed / "Dataset_processed.csv")
df = df[df.well_data_source == 'UCD']
df = df[['well_id','Conductivity','mean_nitrate','area_wt_sagbi', 'Cafo_Population_5miles','Average_ag_area','total_ag_2010','APPROXIMATE LATITUDE', 'APPROXIMATE LONGITUDE']]


df2 = df[(df['Conductivity'] >= .2) & (df['mean_nitrate'] >= 40)]
#get the unique well ids
well_ids = df["well_id"].unique()
# %%
#======================================================================
# Plotting function
#======================================================================
def scatter_twovars(df, xvar = "Cafo_Population_5miles", yvar ="mean_nitrate",
                    xlab = "Animal population", ylab = "Nitrate concentration"):
    
    # scatterplot the nitrate and animal polulation for those well_ids
    plt.scatter(df[xvar], df[yvar])
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()
# %%
scatter_twovars(df2, xvar = "Cafo_Population_5miles", yvar ="mean_nitrate",
                    xlab = "Animal population", ylab = "Nitrate concentration")
# %%
scatter_twovars(df2, xvar = "Conductivity", yvar ="mean_nitrate",
                    xlab = "Conductivity", ylab = "Nitrate concentration")

# %%
scatter_twovars(df2, xvar = "Average_ag_area", yvar ="mean_nitrate",
                    xlab = "Average agricultural area", ylab = "Nitrate concentration")

# %%
scatter_twovars(df2, xvar = "area_wt_sagbi", yvar ="mean_nitrate",
                    xlab = "SAGBI rating", ylab = "Nitrate concentration")


# %%
