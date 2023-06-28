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
def get_aem_resistivity_plot(gdfres, resistivity_above = None, reg=None, conductivity_max_lim=None,remove_highsaline_regions_flag=None,select_subset_lowres_regions=None):
    aemres = gdfres.copy()
    # aemres = gpd.clip(aemres, reg)
    aemres = gpd.sjoin(aemres, reg, op="within")

    aemres = aemres.to_crs(epsg=4326)

    aemres = aemres.dropna(subset=['Resistivity'])
    aemres.Resistivity = 1/aemres.Resistivity

    aemres.loc[aemres.Resistivity > 50, 'Resistivity'] = 50  # Set values greater than 50 to 50

    if remove_highsaline_regions_flag == 1:
        # define the list of SubRegions to exclude
        exclude_subregions = [14, 15, 10, 19,18, 9,6]

        # filter aemres to keep only the rows where Resistivity is >= 10 and SubRegion is not in the exclude_subregions list
        aemres = aemres[(aemres['Resistivity'] >= 8) | (~aemres['SubRegion'].isin(exclude_subregions))]

    if select_subset_lowres_regions == 1:
        # Assign values for different resistivity ranges
        aemres.loc[aemres.Resistivity <= 5, 'Resistivity'] = 1
        aemres.loc[(aemres.Resistivity > 5) & (aemres.Resistivity <= 10), 'Resistivity'] = 2
        aemres.loc[(aemres.Resistivity > 10) & (aemres.Resistivity <= 15), 'Resistivity'] = 3
        aemres.loc[(aemres.Resistivity > 15) & (aemres.Resistivity <= 20), 'Resistivity'] = 4

        # Remove rows where aemres.Resistivity is not 1 or 2
        aemres = aemres[(aemres.Resistivity == 1) | (aemres.Resistivity == 2) | (aemres.Resistivity == 3) | (aemres.Resistivity == 4)]

    # Remove parts with Resistivity > 31
    if resistivity_above is not None:
        aemres = aemres[aemres.Resistivity > resistivity_above]
    
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
aemres_tmp, depth_avg_res = get_aem_resistivity_plot(gdfres = gdfaem, resistivity_above = None, reg = cv_subreg, conductivity_max_lim = None,remove_highsaline_regions_flag=1)
aemres_tmp_wtsalineregion, depth_avg_res = get_aem_resistivity_plot(gdfres = gdfaem, resistivity_above = None, reg = cv_subreg, conductivity_max_lim = None,remove_highsaline_regions_flag=None,select_subset_lowres_regions=None)
aemres_tmp_subsetregion, depth_avg_res = get_aem_resistivity_plot(gdfres = gdfaem, resistivity_above = None, reg = cv_subreg, conductivity_max_lim = None,remove_highsaline_regions_flag=1,select_subset_lowres_regions=1)

#%%
aemres2 = aemres_tmp[['Resistivity', 'geometry', 'HR', 'SubRegion']]
aemres_wtsaline = aemres_tmp_wtsalineregion[['Resistivity', 'geometry', 'HR', 'SubRegion']]
aemres_subset = aemres_tmp_subsetregion[['Resistivity', 'geometry', 'HR', 'SubRegion']]

#%%
aemres2.to_file(config.data_processed / "DAR/DAR_9lyrs.shp", driver="ESRI Shapefile") # export to GeoJSON
aemres_wtsaline.to_file(config.data_processed / "DAR/DAR_9lyrs_wt_saline.shp", driver="ESRI Shapefile") # export to GeoJSON
aemres_subset.to_file(config.data_processed / "DAR/DAR_9lyrs_subsetregions.shp", driver="ESRI Shapefile") # export to GeoJSON

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
def get_darplot(reg = 1, HR = HR, gdf_gwpa_lch = gdf_gwpa_lch):
    i = reg
    # Create the plot
    fig = plt.figure(figsize=(14, 10))
    fig.set_facecolor("white") # set the background color to gray

    out = plt.scatter(
        aemres2['geometry'].x, aemres2['geometry'].y, c=aemres2.Resistivity, 
        s=.7, 
        # cmap=cmap,
        # norm=norm,
        zorder=1
    )

    # Plot the shapefile using geopandas.plot() method
    gdf_gwpa_lch.plot(ax=plt.gca(), facecolor='none', edgecolor='red', linewidth=.7)

    # Plot the shapefile using geopandas.plot() method
    HR.plot(ax=plt.gca(), facecolor='none', edgecolor='orange', linewidth=1)

    # plt.xlabel("Easting (m)")
    # plt.ylabel("Northing (m)")
    cbar = plt.colorbar(out, fraction=0.03, orientation='horizontal')
    cbar.set_label('Depth Average Resistivity', size = 14) 
    plt.xlim(HR.bounds.iloc[i].minx,HR.bounds.iloc[i].maxx)
    plt.ylim(HR.bounds.iloc[i].miny,HR.bounds.iloc[i].maxy)
    plt.axis(False)

#%%
get_darplot(reg = 2, HR = HR, gdf_gwpa_lch = gdf_gwpa_lch)
#%%
get_darplot(reg = 1, HR = HR, gdf_gwpa_lch = gdf_gwpa_lch)
#%%
get_darplot(reg = 3, HR = HR, gdf_gwpa_lch = gdf_gwpa_lch)
#%%
get_darplot(reg = 4, HR = HR, gdf_gwpa_lch = gdf_gwpa_lch)


# %%
# Save figure as PNG
def get_kml_spatial(cv,kk, aemres_tmp,output_name = 'depth_average_resistivity'):
    kk.savefig(config.data_processed / f'{output_name}.png', dpi=300, bbox_inches='tight')

    aemres_tmp = aemres_tmp.to_crs(epsg=4326) # Reproject to WGS84
    # Define image bounds and georeference information
    north = cv.geometry.total_bounds[3]
    south = cv.geometry.total_bounds[1]
    east = cv.geometry.total_bounds[2]
    west = cv.geometry.total_bounds[0]
    width = kk.get_size_inches()[0] * kk.dpi
    height = kk.get_size_inches()[1] * kk.dpi
    georef = f'<GroundOverlay>\n<name>{output_name}</name>\n<Icon>\n<href>file:////Users/szalam/Library/CloudStorage/GoogleDrive-szalam@stanford.edu/Shared drives/GWAttribution/data/processed/{output_name}.png</href>\n</Icon>\n<LatLonBox>\n<north>{north}</north>\n<south>{south}</south>\n<east>{east}</east>\n<west>{west}</west>\n<rotation>0.0</rotation>\n</LatLonBox>\n</GroundOverlay>'

    # Save georeference information to KML file
    with open(config.data_processed /f'{output_name}.kml', 'w') as f:
        f.write(georef)
# %%
get_kml_spatial(cv,depth_avg_res, aemres_tmp,output_name = 'depth_average_resistivity')

