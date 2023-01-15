# %%
from pickle import NONE
import sys
import pandas as pd
# import fiona
# import pyproj
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
# from sklearn.neighbors import BallTree
# from pyproj import CRS

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

def gwdep_array_gdf(gwdp,gwdp_x,gwdp_y):
    arr = gwdp
    # shp = arr.shape
    r = gwdp_x
    c = gwdp_y
    df_res = pd.DataFrame(np.c_[r.ravel(), c.ravel(), arr.ravel()], \
                                    columns=((['x','y','gwdep'])))

    # converting aem data to geodataframe
    gdfres = gpd.GeoDataFrame(
        df_res, geometry=gpd.points_from_xy(df_res.x, df_res.y))

    # setting data coordinate to lat lon
    gdfres = gdfres.set_crs(epsg='4326')

    return gdfres


def get_aem_from_npy(file_loc_interpolated, file_aem_interpolated, aemregion, aemsrc = 'DWR'):
        """
        Get AEM resistivity values as geodataframe. Though the column name for AEM value is referred as
        Resistivity thoughout the model, it will indicate conductivity if the imported data is conductivity. 
        """
        aem_interp = np.load(file_loc_interpolated / f'{file_aem_interpolated}')
        if aemsrc == 'DWR':
            aem_interp_X = np.load(file_loc_interpolated / f'X_region{aemregion}.npy')
            aem_interp_Y = np.load(file_loc_interpolated / f'Y_region{aemregion}.npy')
        if aemsrc == 'ENVGP':
            aem_interp_X = np.load(file_loc_interpolated / f'X.npy')
            aem_interp_Y = np.load(file_loc_interpolated / f'Y.npy')

        arr = aem_interp
        # shp = arr.shape
        r = aem_interp_X
        c = aem_interp_Y
        df_res = pd.DataFrame(np.c_[r.ravel(), c.ravel(), arr.ravel()], \
                                        columns=((['x','y','Resistivity'])))
        
        # converting aem data to geodataframe
        gdfres = gpd.GeoDataFrame(
            df_res, geometry=gpd.points_from_xy(df_res.x, df_res.y))
        
        # Convert aem data coordinate to lat lon
        if aemsrc == 'DWR':
            gdfres = gdfres.set_crs(epsg='3310')
        if aemsrc == 'ENVGP':
            gdfres = gdfres.set_crs(epsg='32611')

        gdfres.to_crs(epsg='4326', inplace=True)

        gdfres = gdfres

        return gdfres

    

def get_agarea_in_wellbuffer(crp_load,tot_well_use = 300,strt_yr_lu = 2008,end_yr_lu = 2020):
    pdf_all = 0
    
    for i in range(0,tot_well_use,1):
        crp_s = crp_load[i].copy()

        crp_ids = [1, 5, 6, 12, 13, 225, 226, 237, 239,
                                            23, 24, 25, 26, 27, 28, 29, 240,
                                            242, 243, 244, 245, 246, 247, 248, 249, 250, 55, 214, 216,219,221, 227, 230, 231, 232, 233,
                                            72, 212, 217,
                                            10,  14, 224, 31,33, 34, 35, 36, 38, 39, 41, 42, 43, 46, 47, 48 ,
                                            49, 50, 51, 52, 53, 54,  56, 57,206,207, 208, 209,213,222, 229,
                                            69,
                                            4, 21, 22, 205, 234, 235, 236,
                                            74, 75, 76,66,77, 223, 68, 210, 220, 67, 70, 71, 204, 211,215,218,
                                            2, 238, 239]
        # first store data for the first year of consideration
        yr = strt_yr_lu
        a_st = 0
        s = pd.Series(crp_s.Crop_id,dtype=np.float32)
        a = crp_s[f"Area_{yr}"][s.isin(crp_ids)].sum()

        # the values indicate total number of grids. converting it to km2. Grid dimension is 30 m
        a = a * 30 * 30 / 1000000
        a_st = [a_st,a]

        a_st.append(a)

        # iterate over other years
        for yr in range(strt_yr_lu+1,end_yr_lu,1):
            s = pd.Series(crp_s.Crop_id,dtype=np.float32)

            a = crp_s[f"Area_{yr}"][s.isin(crp_ids)].sum()
            a_st.append(a)


        b = pd.DataFrame({f'well_{i}': a_st})

        if i==0:
            pdf_all = b
        else:
            pdf_all = pd.concat([pdf_all,b],axis = 1)

    # # remove first row with zero
    pdf_all = pdf_all.iloc[2: , :]
    pdf_all = pdf_all.reset_index()
    pdf_all = pdf_all.drop(['index'], axis=1)


    # # combine the years column
    yrs_lu = pd.DataFrame({'LU_yrs': range(strt_yr_lu,end_yr_lu)})
    pdf_all = pd.concat([yrs_lu,pdf_all],axis = 1)
    pdf_all.index = pdf_all.LU_yrs
    pdf_all = pdf_all.drop(['LU_yrs'], axis=1)
    pdf_all_t = pdf_all.T
    pdf_all_t = pdf_all_t.reset_index()

    # read the well data used for lu extraction
    well_use_f_lu = get_region(config.data_processed / 'Well_buffer_shape/Well_buffer_2mil.shp')
    well_n = pdf_all_t.shape[0]
    well_ids = well_use_f_lu['WELL ID'][:well_n]

    # add column with well ids
    pdf_all_t = pd.concat([well_ids,pdf_all_t],axis = 1)
    ag_area = pdf_all_t.drop(['index'], axis=1)

    return ag_area

# export well buffer shapefile of user given radius (miles)
def get_well_buffer_shape(df,rad_buffer = 2):
    
    if 'APPROXIMATE LONGITUDE' in df:
        # Create a example dataframe with lat and lon columns
        # df = pd.DataFrame({'well_id':df['WELL ID'],'lat':df['APPROXIMATE LATITUDE'], 'lon':df['APPROXIMATE LONGITUDE']})
        df.rename(columns = {'APPROXIMATE LATITUDE': 'lat', 'APPROXIMATE LONGITUDE':'lon'}, inplace = True)
    if 'WELL ID' in df:
        # Create a example dataframe with lat and lon columns
        df.rename(columns = {'WELL ID': 'well_id'}, inplace = True)

    # Convert latitude and longitude columns to shapely Points
    geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]

    # Create a GeoDataFrame
    crs = {'init': 'epsg:4326'} # CRS stands for Coordinate Reference System, this line specify the system
    gdf2 = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)

    # # Convert the CRS of the GeoDataFrame to a projected CRS (e.g. UTM zone 10N)
    gdf_buffer = gdf2.to_crs({'init': 'epsg:32610'})

    # # Create a buffer of 2 miles in the projected CRS
    gdf_buffer.geometry = gdf_buffer.buffer(1609.345 * rad_buffer)

    # # Convert the CRS back to WGS 84
    gdf_buffer = gdf_buffer.to_crs({'init': 'epsg:4326'})

    return gdf_buffer
    

def get_luids():

   

    data = {
        # Water = water(83), wetlands(87), Aquaculture(92), Open Water(111), Perreniel Ice / Snow (112)
        1 : [83, 87, 92, 111, 112], 
        # Urban = developed high intensity(124), developed medium intensity(123)
        2 : [124, 123], 
        # Native = grassland/pasture(176), Forest(63), Shrubs(64), barren(65, 131), Clover/Wildflowers(58)
        # Forests (141 - 143), Shrubland (152), Woody Wetlands (190), Herbaceous wetlands (195)
        3 : [176,63,64, 65, 131,58, 141, 142, 143, 152, 190, 195], 
        # Orchards, groves, vineyards = 
        4 : [],
        # Pasture / hay = other hay / non alfalfa (37)
        5 : [37],
        # Row Crops = corn (1), soybeans (5),Sunflower(6) sweet corn (12), pop corn (13), double winter/corn (225), 
        # double oats/corn(226), double barley/corn(237), double corn / soybeans
        6 : [1, 5, 6, 12, 13, 225, 226, 237, 239] ,
        # Small Grains = Spring wheat (23), winter wheat (24), other small grains (25), winter wheat / soybeans (26), 
        # rye (27), oats (28), Millet(29), dbl soybeans/oats(240)
        7 : [23, 24, 25, 26, 27, 28, 29, 240] ,
        # Idle/fallow = Sod/Grass Seed (59), Fallow/Idle Cropland(61), 
        8 : [59,61],
        # Truck, nursery, and berry crops = 
        # Blueberries (242), Cabbage(243), Cauliflower(244), celery (245), radishes (246), Turnips(247)
        # Eggplants (249), Cranberries (250), Caneberries (55), Brocolli (214), Peppers(216), 
        # Greens(219), Strawberries (221), Lettuce (227), Double Lettuce/Grain (230 - 233)
        9 : [242, 243, 244, 245, 246, 247, 248, 249, 250, 55, 214, 216,219,221, 227, 230, 231, 232, 233], 

        # Citrus and subtropical = Citrus(72), Oranges (212), Pommegranates(217)
        10 : [72, 212, 217] ,

        # Field Crops = 
        # Peanuts(10),Mint (14),Canola (31),  Vetch(224),  Safflower(33) , RapeSeed(34), 
        # Mustard(35) Alfalfa (36),Camelina (38), Buckwheat (39), Sugarbeet (41), Dry beans (42), Potaoes (43)
        # Sweet potatoes(46), Misc Vegs & Fruits (47), Cucumbers(50)
        # Chick Peas(51),Lentils(52),Peas(53),Tomatoes(54)Hops(56),Herbs(57),Carrots(206),
        # Asparagus(207),Garlic(208), Cantaloupes(209), Honeydew Melons (213), Squash(222), Pumpkins(229), 

        11 : [10,  14, 224, 31,33, 34, 35, 36, 38, 39, 41, 42, 43, 46, 47, 48 ,
              49, 50, 51, 52, 53, 54,  56, 57,206,207, 208, 209,213,222, 229] ,

        # Vineyards = Grapes(69)
        12 : [69],
        # Pasture = Switchgrass(60)
        13 : [60],
        # Grain and hay = Sorghum(4), barley (21), Durham wheat (22), Triticale (205), 
        # Dbl grain / sorghum (234 - 236), Dbl 
        14 : [4, 21, 22, 205, 234, 235, 236],
        # livestock feedlots, diaries, poultry farms = 
        15 : [],

        # Deciduous fruits and nuts = Pecans(74), Almonds(75), 
        # Walnuts(76), Cherries (66), Pears(77), Apricots (223), Apples (68), Christmas Trees(70)
        # Prunes (210), Plums (220), Peaches(67), Other Tree Crops (71), Pistachios(204), 
        # Olives(211), Nectarines(218), Avocado (215)
        16 : [74, 75, 76,66,77, 223, 68, 210, 220, 67, 70, 71, 204, 211,215,218],

        # Rice = Rice(3)
        17 : [3],
        # Cotton = Cotton (2) , Dbl grain / cotton (238-239)
        18 : [2, 238, 239], 
        # Developed = Developed low intensity (122) developed open space(121)
        19 : [122, 121],
        # Cropland and Pasture
        20 : [],
        # Cropland = Other crops (44)
        21 : [44], 
        # Irrigated row and field crops = Woody Wetlands (190), Herbaceous wetlands(195)
        22 : [] # [190, 195] 
        }

    data[23] = list(range(256))

    return data

def get_luid_names():


    data = {
        # Water = water(83), wetlands(87), Aquaculture(92), Open Water(111), Perreniel Ice / Snow (112)
        1 : ["Water"], 
        # Urban = developed high intensity(124), developed medium intensity(123)
        2 : ["Urban"], 
        # Native = grassland/pasture(176), Forest(63), Shrubs(64), barren(65, 131), Clover/Wildflowers(58)
        # Forests (141 - 143), Shrubland (152), Woody Wetlands (190), Herbaceous wetlands (195)
        3 : ["Forests"], 
        # Orchards, groves, vineyards = 
        4 : [""],
        # Pasture / hay = other hay / non alfalfa (37)
        5 : ["Pasture"],
        # Row Crops = corn (1), soybeans (5),Sunflower(6) sweet corn (12), pop corn (13), double winter/corn (225), 
        # double oats/corn(226), double barley/corn(237), double corn / soybeans
        6 : ["Row_crops"] ,
        # Small Grains = Spring wheat (23), winter wheat (24), other small grains (25), winter wheat / soybeans (26), 
        # rye (27), oats (28), Millet(29), dbl soybeans/oats(240)
        7 : ["Small_grains"] ,
        # Idle/fallow = Sod/Grass Seed (59), Fallow/Idle Cropland(61), 
        8 : ["Idle"],
        # Truck, nursery, and berry crops = 
        # Blueberries (242), Cabbage(243), Cauliflower(244), celery (245), radishes (246), Turnips(247)
        # Eggplants (249), Cranberries (250), Caneberries (55), Brocolli (214), Peppers(216), 
        # Greens(219), Strawberries (221), Lettuce (227), Double Lettuce/Grain (230 - 233)
        9 : ["Truck_nursery_berry"], 

        # Citrus and subtropical = Citrus(72), Oranges (212), Pommegranates(217)
        10 : ["Citrus_subtropical"] ,

        # Field Crops = 
        # Peanuts(10),Mint (14),Canola (31),  Vetch(224),  Safflower(33) , RapeSeed(34), 
        # Mustard(35) Alfalfa (36),Camelina (38), Buckwheat (39), Sugarbeet (41), Dry beans (42), Potaoes (43)
        # Sweet potatoes(46), Misc Vegs & Fruits (47), Cucumbers(50)
        # Chick Peas(51),Lentils(52),Peas(53),Tomatoes(54)Hops(56),Herbs(57),Carrots(206),
        # Asparagus(207),Garlic(208), Cantaloupes(209), Honeydew Melons (213), Squash(222), Pumpkins(229), 

        11 : ["Field_crops"] ,

        # Vineyards = Grapes(69)
        12 : ["Vineyards"],
        # Pasture = Switchgrass(60)
        13 : ["Pasture"],
        # Grain and hay = Sorghum(4), barley (21), Durham wheat (22), Triticale (205), 
        # Dbl grain / sorghum (234 - 236), Dbl 
        14 : ["Grain_hay"],
        # livestock feedlots, diaries, poultry farms = 
        15 : [""],

        # Deciduous fruits and nuts = Pecans(74), Almonds(75), 
        # Walnuts(76), Cherries (66), Pears(77), Apricots (223), Apples (68), Christmas Trees(70)
        # Prunes (210), Plums (220), Peaches(67), Other Tree Crops (71), Pistachios(204), 
        # Olives(211), Nectarines(218), Avocado (215)
        16 : ["Deciduous_fruits_nuts"],

        # Rice = Rice(3)
        17 : ["Rice"],
        # Cotton = Cotton (2) , Dbl grain / cotton (238-239)
        18 : ["Cotton"], 
        # Developed = Developed low intensity (122) developed open space(121)
        19 : ["Developed"],
        # Cropland and Pasture
        20 : [""],
        # Cropland = Other crops (44)
        21 : ["Other_crops"], 
        # Irrigated row and field crops = Woody Wetlands (190), Herbaceous wetlands(195)
        22 : [""], # ["190", "195"]
        23 : ["All"] 
        }

    return data

def lu_cdl_filname():

    filname_lu = {
        2007 : ["CDL_2007_clip_20230113012728_1621246391.tif"], 
        2008 : ["CDL_2008_clip_20230113012728_1621246391.tif"],
        2009 : ["CDL_2009_clip_20230113012728_1621246391.tif"],
        2010 : ["CDL_2010_clip_20230113012728_1621246391.tif"],
        2011 : ["CDL_2011_clip_20230113012728_1621246391.tif"],
        2012 : ["CDL_2012_clip_20230113130339_1286798411.tif"],
        2013 : ["CDL_2013_clip_20230113130339_1286798411.tif"],
        2014 : ["CDL_2014_clip_20230113130339_1286798411.tif"],
        2015 : ["CDL_2015_clip_20230113130339_1286798411.tif"],
        2016 : ["CDL_2016_clip_20230113132411_919581983.tif"],
        2017 : ["CDL_2017_clip_20230113132411_919581983.tif"],
        2018 : ["CDL_2018_clip_20230113132411_919581983.tif"],
        2019 : ["CDL_2019_clip_20230113134140_113242721.tif"],
        2020 : ["CDL_2020_clip_20230113134140_113242721.tif"],
        2021 : ["CDL_2021_clip_20230113134140_113242721.tif"]
    }

    return filname_lu