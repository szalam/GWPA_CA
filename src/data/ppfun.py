# %%
from pickle import NONE
import sys
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from sklearn.neighbors import BallTree

sys.path.insert(0,'src')
import config
"""
Read raw nitrate dataset, separate for given shapefile, and given range of years.
"""
# %%
def get_polut_df(file_sel):

    """
    Read data csv and separate specific chemicals.
    Args:
    chemical: chemical data that needs to be separated
    """
    # Read nitrate data from csv
    c = pd.read_csv(file_sel)

    # convert the 'Date' column to datetime format
    c['DATE']= pd.to_datetime(c['DATE'])

    return c

def get_polut_stat(c_df, uniq = 0, uniq_col = 'WELL ID', stat_sel = 'max'):
    
    # Drop duplicate rows with name well name
    if uniq == 1:
        c2 = c_df.drop_duplicates(subset=[uniq_col])

    if uniq == 0:
        if stat_sel == 'max':
            c_stat = c_df.groupby('WELL ID').max()['RESULT']
        if stat_sel == 'min':
            c_stat = c_df.groupby('WELL ID').min()['RESULT']
        if stat_sel == 'mean':
            c_stat = c_df.groupby('WELL ID').mean()['RESULT']
        
        c_stat = pd.DataFrame(c_stat)
        c_stat.rename({'RESULT': 'VALUE'}, axis=1, inplace=True)
        c_stat.index.names = ['id']

        c_stat['WELL ID'] = c_stat.index
        c2 =  c_stat.merge(c_df, on = 'WELL ID', how = 'left')
        c2.rename({'RESULT': 'RESULT_Dont_Use'}, axis=1, inplace=True)
        c2 = c2.drop_duplicates(subset=[uniq_col])
    
    return c2

def get_polut_scatter(cmax_c,x_val,yaxis_lim = None):
    fig, ax = plt.subplots(figsize=(7, 7))
    
    sc = ax.scatter(cmax_c[x_val],cmax_c.VALUE)
    # plt.legend(loc="upper left")
    plt.xlabel(x_val, fontsize=20)
    plt.ylabel('Polut concentration', fontsize=20)
    # plt.legend(fontsize=17) 
    plt.tick_params(axis='both', which='major', labelsize=17)
    if yaxis_lim is not None:
            plt.ylim([0,yaxis_lim])
    plt.show()

def df_to_gpd(c_df, lat_col_name = 'APPROXIMATE LATITUDE', lon_col_name = 'APPROXIMATE LONGITUDE'):
    """
    Convert the pandas dataframe to geodataframe
    """
    # Convert pollutant dataframe data to geodataframe
    # creating a geometry column 
    geometry = [Point(xy) for xy in zip(c_df[lon_col_name], c_df[lat_col_name])]
    # Coordinate reference system : WGS84
    crs = {'init': 'epsg:4326'}
    # Creating a Geographic data frame 
    gdf_pol = gpd.GeoDataFrame(c_df, crs=crs, geometry=geometry)

    return gdf_pol

# %%
def get_polut_for_gwbasin(c_df, gw_basin = "SAN JOAQUIN VALLEY - KAWEAH (5-22.11)"):
    """
    Separate contaminant data for specific groundwater basin (gw_basin)
    Args:
    c_df = dataframe of contaminant
    gw_basin = groundwater basin for which contaminant data needs to be separated. Default to Kaweah
    """
    c = c_df.loc[c_df['GW_BASIN_NAME'] == gw_basin]

    return c

# %%
def get_polut_for_county(c_df, county = "KERN"):
    """
    Separate contaminant data for specific groundwater basin (gw_basin)
    Args:
    c_df = dataframe of contaminant
    gw_basin = groundwater basin for which contaminant data needs to be separated. Default to Kaweah
    """
    c = c_df.loc[c_df['COUNTY'] == county]
    
    return c

#%%
def get_df_for_dates(c_df,start_date = '2000-01-01', end_date = '2021-12-31'):
    """
    Separate data for selected time span
    Args:
    c_df = pollutant dataframe
    start_date, end_date = range between which data separated
    """
    # Ids within given range
    sel_ids = (c_df['DATE'] > start_date) & (c_df['DATE'] <= end_date)
    
    # Separate data within the date ranges
    c = c_df.loc[sel_ids]

    return(c)

#%%
def get_region(file_domain):
    """
    Read study region shape and convert to lat lon
    """
    reg = gpd.read_file(file_domain)
    reg.to_crs(epsg='4326', inplace=True)
    return reg

#%%
def point_clip(file_sel = None, pt_dt = None, reg = None):
    """
    Clip geolocated points using shapefile
    """

    if pt_dt is not None:
        pt = pt_dt

    if file_sel is not None:
        pt = gpd.read_file(file_sel)
        pt.to_crs(epsg='4326', inplace=True)

    if reg is not None:
        pt = gpd.clip(pt, reg) #clip wpa using kw shape
    
    return pt

#%% 
def plt_domain(gdf_pol, mcl = None , region = None, cafo_shp = None, gwpa = None, aem = None, well = None, welltype = None, polut_factor = None, c_name = 'Nitrate'):
    """
    Plot water quality measurement locations along with other shapes

    Args:
    gdf: input geodataframe of the pollutant
    mcl: maximum contaminant level above which to plot for selected pollutant. if None, it will plot all
    region: domain to plot
    cafo_shp: CAFO point geopandas dataframe
    gwpa: GWPA geopandas dataframe
    well: well geopandas dataframe
    welltype: spefic well type to plot. if None, it will plot all
    polut_factor: a factor multiplied to the marker size of the pollutant magnitude for size adjustment
    """
    # gdf_pol = df_to_gpd(c_df)
 
    if region is not None:
        gdf_pol = gpd.clip(gdf_pol, region) 

    # plot
    fig, ax = plt.subplots(1,1,figsize = (10,10))
    
    if mcl is not None:
        gdf_pol = gdf_pol[gdf_pol.VALUE > mcl]
        
    if polut_factor is None:
        gdf_pol.plot(ax = ax, label=c_name, color = 'b', markersize = .5, zorder=10, alpha =.7)
        
    if polut_factor is not None:
        siz = gdf_pol.VALUE
        gdf_pol.plot(ax = ax, label=c_name, color = 'b', markersize = siz*polut_factor, zorder=2, alpha =.7)
    
    if region is not None:
        region.plot( ax = ax, label = 'Region',facecolor = 'none' ,edgecolor = 'grey', lw = .5, zorder=1, alpha = .8)
    
    if cafo_shp is not None:
        cafo_shp.plot( ax = ax, label = 'Cafo',edgecolor = 'grey', lw = .5, zorder=1, alpha = .8)
    
    if gwpa is not None:
        gwpa.plot( ax = ax, label = 'GWPA',facecolor = 'gray', edgecolor = 'black', lw = .5, zorder=1, alpha = .3)
    
    if aem is not None:
        aem.plot( ax = ax, label = 'AEM',facecolor = 'none', edgecolor = 'brown', lw = .5, zorder=3, alpha = .8)
    
    if well is not None:
        if welltype is None:
            well.plot( ax = ax, label = 'Well',edgecolor = 'green', lw = .5, zorder=1, alpha = .8)
        if welltype is not None:
            well = well.loc[well['well_use'] == welltype]
            well.plot( ax = ax, label = 'Well',edgecolor = 'green', lw = .5, zorder=1, alpha = .8)

    plt.axis(False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, fontsize=12, loc='upper right')



def get_nearest(src_points, candidates, k_neighbors=1):
    """Find nearest neighbors for all source points from a set of candidate points"""

    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15, metric='haversine')

    # Find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)

    # Transpose to get distances and indices into arrays
    distances = distances.transpose()
    indices = indices.transpose()

    # Get closest indices and distances (i.e. array at index 0)
    # note: for the second closest points, you would take index 1, etc.
    closest = indices[0]
    closest_dist = distances[0]

    # Return indices and distances
    return (closest, closest_dist)


def nearest_neighbor(left_gdf, right_gdf, return_dist=False):
    """
    For each point in left_gdf, find closest point in right GeoDataFrame and return them.

    NOTICE: Assumes that the input Points are in WGS84 projection (lat/lon).
    """

    left_geom_col = left_gdf.geometry.name
    right_geom_col = right_gdf.geometry.name

    # Ensure that index in right gdf is formed of sequential numbers
    right = right_gdf.copy().reset_index(drop=True)

    # Parse coordinates from points and insert them into a numpy array as RADIANS
    left_radians = np.array(left_gdf[left_geom_col].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())
    right_radians = np.array(right[right_geom_col].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())

    # Find the nearest points
    # -----------------------
    # closest ==> index in right_gdf that corresponds to the closest point
    # dist ==> distance between the nearest neighbors (in meters)

    closest, dist = get_nearest(src_points=left_radians, candidates=right_radians)

    # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame
    closest_points = right.loc[closest]

    # Ensure that the index corresponds the one in left_gdf
    closest_points = closest_points.reset_index(drop=True)

    # Add distance if requested
    if return_dist:
        # Convert to meters from radians
        earth_radius = 6371000  # meters
        closest_points['distance'] = dist * earth_radius

    return closest_points