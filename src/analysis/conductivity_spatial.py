#%%
import sys
sys.path.insert(0,'src')
sys.path.insert(0,'src/data')
import config
import fiona
import warnings
import pandas as pd
import ppfun as dp
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from geopandas import GeoDataFrame
from pyproj import CRS
from tqdm import tqdm
import matplotlib.pyplot as plt
import get_infodata as gi
import matplotlib as mpl

from contextlib import redirect_stdout

# file location
file_polut = config.data_raw / "nitrate_data/UCDNitrateData.csv"
file_aem = config.data_processed / 'AEM'

#==================== User Input requred ==========================================
aem_src        = 'DWR'          # options: DWR, ENVGP
well_src       = 'GAMA'          # options: UCD, GAMA
aem_regions = [4, 5, 6, 7] # list of all regions
# aem_reg2       = 4              # use only if two regions are worked with
aem_lyr_lim    = 9              # options: 9, 8. For DWR use 9,3,20, for ENVGP use 8
aem_value_type = 'conductivity' # options: resistivity, conductivity
aem_stat       = 'mean'         # options: mean, min
rad_buffer     = 2              # well buffer radius in miles
buffer_flag    = 0              # flag 1: use existing buffer shapefile; 0: create buffer
cond_thresh    = 0.1
#==================================================================================
#============================ Import AEM data ===============================

def aem_info(file_aem, aem_src, aem_value_type, aem_lyr_lim, aem_reg):
    aem_fil_loc = file_aem / aem_src / aem_value_type
    if aem_src == 'DWR':
        interpolated_aem_file = f'{aem_value_type}_lyrs_{aem_lyr_lim}_reg{aem_reg}.npy'
    elif aem_src == 'ENVGP':
        interpolated_aem_file = f'{aem_value_type}_lyrs_{aem_lyr_lim}.npy'
    return aem_fil_loc, interpolated_aem_file

# # call the function
# aem_fil_loc1, interpolated_aem_file1 = aem_info(file_aem, aem_src, aem_value_type, aem_lyr_lim, aem_reg)
# aem_fil_loc2, interpolated_aem_file2 = aem_info(file_aem, aem_src, aem_value_type, aem_lyr_lim, aem_reg2)


# Create a list of arguments for the get_aem_from_npy function
aem_args = []
for aem_region in aem_regions:
    aem_fil_loc, interpolated_aem_file = aem_info(file_aem, aem_src, aem_value_type, aem_lyr_lim, aem_region)
    aem_args.append({
        'file_loc_interpolated': aem_fil_loc, 
        'file_aem_interpolated': interpolated_aem_file, 
        'aemregion': aem_region, 
        'aemsrc': aem_src
    })

#%%
# Use a list comprehension to apply the get_aem_from_npy function to each set of arguments
gdf_aem_list = [dp.get_aem_from_npy(**args) for args in aem_args]

# Concatenate the dataframes into one
gdfaem = pd.concat(gdf_aem_list)


#%%
def get_aem_conductivity_plot(gdfres, reg=None, conductivity_max_lim=None):
    aemres = gdfres.copy()
    aemres = gpd.clip(aemres, reg)
    aemres = aemres.dropna(subset=['Resistivity'])
    
    # Remove parts with Resistivity > 31
    aemres = aemres[aemres.Resistivity < .1]
    
    # Define Resistivity value ranges and associated colors
    # ranges = [0, .01,.02,.03,.04,.05,.06,.07,.08,.09]
    # colors = ['purple', 'blue', 'deepskyblue', 'green', 'yellow', 'orange', 'red', 'saddlebrown', 'gray', 'pink']
    
    # # Create a colormap using the defined colors and ranges
    # cmap = mpl.colors.ListedColormap(colors)
    # norm = mpl.colors.BoundaryNorm(ranges, cmap.N)
    
    # Create the plot
    fig = plt.figure(figsize=(7, 10))
    out = plt.scatter(
        aemres['geometry'].x, aemres['geometry'].y, c=aemres.Resistivity, 
        s=.001, 
        # cmap=cmap,
        # norm=norm,
        zorder=1
    )
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    cbar = plt.colorbar(out, fraction=0.03)
    cbar.set_label('Depth Average Conductivity (~30)') 
    plt.axis(False)
# %%
cv = gi.get_region(reg_sel = 'cv')
get_aem_conductivity_plot(gdfres = gdfaem, reg = cv, conductivity_max_lim = None)
# %%
