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
aem_lyr_lim    = 9              # options: 9,6,4, 8. For DWR use 9,3,20, for ENVGP use 8. Depths, 6: 16.982, 4: 9.8129
aem_value_type = 'conductivity' # options: resistivity, conductivity
aem_stat       = 'mean'         # options: mean, min
rad_buffer     = 2              # well buffer radius in miles
buffer_flag    = 0              # flag 1: use existing buffer shapefile; 0: create buffer
cond_thresh    = 0.1

# read in city shapefile
cv_subreg = gi.get_region(reg_sel = 'cv_subreg')
# cv_subreg = cv_subreg[cv_subreg.SubRegion != 14]
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
# plot conductivity
def get_aem_conductivity_plot(gdfres, cond_thresh = None, reg=None, conductivity_max_lim=None):
    aemres = gdfres.copy()
    aemres = gpd.clip(aemres, reg)
    aemres = aemres.dropna(subset=['Resistivity'])
    
    # Remove parts with Resistivity > 31
    if cond_thresh is not None:
        aemres = aemres[aemres.Resistivity < cond_thresh]
    
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

    return aemres, fig

# Plot Resistivity
def get_aem_resistivity_plot(gdfres, resistivity_above = None, reg=None, conductivity_max_lim=None):
    aemres = gdfres.copy()
    # aemres = gpd.clip(aemres, reg)
    aemres = gpd.sjoin(aemres, reg, op="within")

    aemres = aemres.to_crs(epsg=4326)

    aemres = aemres.dropna(subset=['Resistivity'])
    aemres.Resistivity = 1/aemres.Resistivity

    # aemres.loc[aemres.Resistivity > 50, 'Resistivity'] = 50  # Set values greater than 50 to 50

    # define the list of SubRegions to exclude
    exclude_subregions = [14, 15, 10, 19,18, 9,6]

    # filter aemres to keep only the rows where Resistivity is >= 10 and SubRegion is not in the exclude_subregions list
    aemres = aemres[(aemres['Resistivity'] >= 8) | (~aemres['SubRegion'].isin(exclude_subregions))]


    # Remove parts with Resistivity > 31
    if resistivity_above is not None:
        aemres = aemres[aemres.Resistivity > resistivity_above]
    
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
    cbar.set_label('Depth Average Resistivity (~10)') 
    plt.axis(False)

    return aemres, fig

# %%
cv = gi.get_region(reg_sel = 'cv')
# get_aem_resistivity_plot(gdfres = gdfaem, resistivity_above = 1, reg = cv_subreg, conductivity_max_lim = None)
aemres_tmp, depth_avg_res = get_aem_resistivity_plot(gdfres = gdfaem, resistivity_above = 1, reg = cv_subreg, conductivity_max_lim = None)

#%%
aemres2 = aemres_tmp[['Resistivity', 'geometry', 'HR', 'SubRegion']]

#%%

# Read the shapefile from Google Drive
shp_path = "CV_subregion.shp"
gdf_hr = gpd.read_file(config.data_raw / 'shapefile/cv_subregion' / shp_path)
gdf_hr = gdf_hr.to_crs('EPSG:4326')

gdf_hr.loc[(gdf_hr['SubRegion'] >= 1) & (gdf_hr['SubRegion'] <= 7), 'HR'] = 'SC'
gdf_hr.loc[(gdf_hr['SubRegion'] >= 8) & (gdf_hr['SubRegion'] <= 8), 'HR'] = 'EB'
gdf_hr.loc[(gdf_hr['SubRegion'] >= 9) & (gdf_hr['SubRegion'] <= 9), 'HR'] = 'DL'
gdf_hr.loc[(gdf_hr['SubRegion'] >= 10) & (gdf_hr['SubRegion'] <= 13), 'HR'] = 'SJ'
gdf_hr.loc[(gdf_hr['SubRegion'] >= 14) & (gdf_hr['SubRegion'] <= 21), 'HR'] = 'TL'


# dissolve by 'HR' column and aggregate the 'area_sqkm' column using 'sum'
dissolved = gdf_hr.dissolve(by='HR', aggfunc='sum')

# reset the index to convert the multi-index back to a regular index
HR = dissolved.reset_index()

#%%
# Read the shapefile from Google Drive
shp_path = "gwpa_leaching.shp"
gdf_gwpa_lch = gpd.read_file(config.data_processed / 'kml' / shp_path)
gdf_gwpa_lch = gdf_gwpa_lch.to_crs('EPSG:4326')

#%%
# Perform a spatial join to get the points that fall within the gdf_gwpa_lch polygons
aemres2_within_gdf_gwpa_lch = gpd.sjoin(aemres2, gdf_gwpa_lch, how='inner', op='within')

# Group points by gdf_gwpa_lch index and calculate mean resistivity for each group
mean_resistivity_per_row = aemres2_within_gdf_gwpa_lch.groupby('index_right')['Resistivity'].mean()

# Generate bin edges with an interval of 10
bin_edges = np.arange(0, 110, 5)

# Plot the histogram
plt.hist(mean_resistivity_per_row, bins=bin_edges, edgecolor='black', color='orange')
plt.xlabel('Mean depth average resistivity', size =14)
plt.ylabel('Frequency', size = 14)
plt.xlim(0, 100)
plt.title('Histogram of mean resistivity values within leaching GWPA')
plt.show()


# %%
bin_edges = np.arange(0, 110, 5)

# Plot the histogram
plt.hist(aemres2['Resistivity'], bins=bin_edges, edgecolor='black', color='orange')
plt.xlabel('Depth average resistivity', size =14)
plt.ylabel('Frequency', size =14)
plt.xlim(0, 100)
plt.title('Histogram of resistivity in Central Valley')
plt.show()
# %%

# Separate GWPA for different HR and then plot histogram
HR_sel = 'TL'
# Make sure both GeoDataFrames have the same CRS
assert HR.crs == gdf_gwpa_lch.crs, "GeoDataFrames must have the same CRS"

# Get the geometry corresponding to the 'SC' value in the HR GeoDataFrame
hr_geometry = HR.loc[HR['HR'] == HR_sel, 'geometry'].iloc[0]

# Create a new GeoDataFrame with only the 'SC' geometry
hr_gdf = gpd.GeoDataFrame([{'HR': HR_sel, 'geometry': hr_geometry}], crs=HR.crs)

# Perform a spatial join between the gdf_gwpa_lch GeoDataFrame and the 'SC' geometry
gdf_gwpa_lch_within_hr = gpd.sjoin(gdf_gwpa_lch, hr_gdf, op='within', how='inner')

# Reset the index of the resulting GeoDataFrame
gdf_gwpa_lch_within_hr.reset_index(drop=True, inplace=True)

# Drop the 'index_right' and 'HR' columns from the gdf_gwpa_lch_within_hr GeoDataFrame
gdf_gwpa_lch_within_hr = gdf_gwpa_lch_within_hr.drop(columns=['index_right', 'HR'])

# Perform a spatial join to get the points that fall within the gdf_gwpa_lch polygons
aemres2_within_gdf_gwpa_lch_hr = gpd.sjoin(aemres2, gdf_gwpa_lch_within_hr, how='inner', op='within')

# Group points by gdf_gwpa_lch index and calculate mean resistivity for each group
mean_resistivity_per_row_hr = aemres2_within_gdf_gwpa_lch_hr.groupby('index_right')['Resistivity'].mean()

# Generate bin edges with an interval of 10
bin_edges = np.arange(0, 110, 5)

# Plot the histogram
plt.hist(mean_resistivity_per_row_hr, bins=bin_edges, edgecolor='black', color='orange')
plt.xlabel('Mean depth average resistivity', size =14)
plt.ylabel('Frequency', size = 14)
plt.xlim(0, 100)
plt.title('Histogram of mean resistivity values within leaching GWPA')
plt.show()


# %%
bin_edges = np.arange(0, 110, 5)

# Plot the histogram
plt.hist(aemres2['Resistivity'], bins=bin_edges, edgecolor='black', color='orange')
plt.xlabel('Depth average resistivity', size =14)
plt.ylabel('Frequency', size =14)
plt.xlim(0, 100)
plt.title('Histogram of resistivity in Central Valley')
plt.show()
# %%
# Export the GeoDataFrame to a shapefile
HR.to_file(config.data_raw / "shapefile/HR_cv.shp")
# %%
