#========================================================================================
# This script calculates the thickness of well buffer above a threhold conductivity value within well buffers.
#========================================================================================
#%%
import sys
sys.path.insert(0,'src')
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
from contextlib import redirect_stdout

def get_thickness_well_buffer(well_src = 'UCD',cond_thresh= 0.15, aem_lyr_lim = 9,rad_buffer = 2, gama_old_new = 2):
    # file location
    file_polut = config.data_raw / "nitrate_data/UCDNitrateData.csv"
    file_aem = config.data_processed / 'AEM'

    #==================== User Input requred ==========================================
    aem_src        = 'DWR'          # options: DWR, ENVGP
    # well_src       = 'UCD'          # options: UCD, GAMA
    aem_regions = [4, 5, 6, 7] # list of all regions
    aem_value_type = 'conductivity' # options: resistivity, conductivity
    aem_stat       = 'mean'         # options: mean, min
    # rad_buffer     = 2              # well buffer radius in miles
    buffer_flag    = 0              # flag 1: use existing buffer shapefile; 0: create buffer
    # cond_thresh    = 0.15           # [0.01, 0.02, 0.03, 0.05, 0.07, 0.08, 0.1, 0.15, 0.18, 0.2]
    # aem_lyr_lim    = 9              # options: 9, 8. For DWR use 9,3,20, for ENVGP use 8
    gama_old_new   = 2              # 1: old dataset, 2: lates2 dataset
    #==================================================================================


    #=========================== Import water quality data ==========================
    if well_src == 'GAMA':
        if gama_old_new == 1:
            file_polut = config.data_gama_all / 'CENTRALVALLEY_NO3N_GAMA.csv'
            df= dp.get_polut_df(file_sel = file_polut)
            df.rename(columns = {'GM_WELL_ID':'WELL ID', 'GM_LATITUDE':'APPROXIMATE LATITUDE', 'GM_LONGITUDE':'APPROXIMATE LONGITUDE', 'GM_CHEMICAL_VVL': 'CHEMICAL', 'GM_RESULT': 'RESULT','GM_WELL_CATEGORY':'DATASET_CAT','GM_SAMP_COLLECTION_DATE':'DATE'}, inplace = True)
            df['DATE']= pd.to_datetime(df['DATE'])

        if gama_old_new == 2:
            file_polut = config.data_processed / "well_stats/gamanitrate_latest_stats.csv"
            df= pd.read_csv(file_polut)
            df.rename(columns = {'well_id':'WELL ID','well_type':'DATASET_CAT'}, inplace = True)

    if well_src == 'UCD':
        # file location
        file_polut = config.data_raw / "nitrate_data/UCDNitrateData.csv"

        # Read nitrate data
        df = dp.get_polut_df(file_sel = file_polut)

    # Group the DataFrame by the 'WELL ID' column and take the first values for other selected columns
    df = df.groupby('WELL ID')['APPROXIMATE LATITUDE', 'APPROXIMATE LONGITUDE',].apply(lambda x: x.iloc[0]).reset_index()

    #============================ Import AEM data ===============================
    thick_loc = 'thickness_abov_threshold_cond'
    def aem_info(file_aem, aem_src, aem_value_type, aem_lyr_lim, aem_reg,cond_thresh):
        aem_fil_loc = file_aem / aem_src / aem_value_type / thick_loc
        if aem_src == 'DWR':
            interpolated_aem_file = f'lyr_thickness_above_threshold_lyrs_{aem_lyr_lim}_cond_{round(cond_thresh*100)}_reg{aem_reg}.npy'
        elif aem_src == 'ENVGP':
            interpolated_aem_file = f'lyr_thickness_above_threshold_lyrs_{aem_lyr_lim}_cond_{round(cond_thresh*100)}_reg{aem_reg}.npy'
        return aem_fil_loc, interpolated_aem_file
   
    # # call the function
    # aem_fil_loc1, interpolated_aem_file1 = aem_info(file_aem, aem_src, aem_value_type, aem_lyr_lim, aem_reg)
    # aem_fil_loc2, interpolated_aem_file2 = aem_info(file_aem, aem_src, aem_value_type, aem_lyr_lim, aem_reg2)


    # Create a list of arguments for the get_aem_from_npy function
    aem_args = []
    for aem_region in aem_regions:
        aem_fil_loc, interpolated_aem_file = aem_info(file_aem, aem_src, aem_value_type, aem_lyr_lim, aem_region,cond_thresh)
        aem_args.append({
            'file_loc_interpolated': aem_fil_loc, 
            'file_aem_interpolated': interpolated_aem_file, 
            'aemregion': aem_region, 
            'aemsrc': aem_src
        })

    
    # Use a list comprehension to apply the get_aem_from_npy function to each set of arguments
    gdf_aem_list = [dp.get_aem_from_npy(**args) for args in aem_args]

    # Concatenate the dataframes into one
    gdfaem = pd.concat(gdf_aem_list)

    #======================== Get boundary of AEM data ========================
    # converting point resistivity data to polygon mask
    gdftmp = gdfaem.copy()
    gdftmp=gdftmp.dropna(subset=['Resistivity'])

    gdftmp['id_1'] = 1
    gdfmask = gdftmp.dissolve(by="id_1")
    gdfmask["geometry"] = gdfmask["geometry"].convex_hull

    gdfaem_boundary = gdfmask

    
    if buffer_flag == 1:
        gdf_wellbuffer = pd.read_pickle(config.data_processed / f'Well_buffer_shape' / f"{well_src}_buffers_{(rad_buffer)}mile.pkl")

    if buffer_flag != 1:
        # Creating buffer around wells
        gdf_wellbuffer_all = dp.get_well_buffer_shape(df,rad_buffer = rad_buffer) 

        # Perform  clip on the well data based on selected boundary
        gdf_wellbuffer = gpd.sjoin(gdf_wellbuffer_all, gdfaem_boundary, how='inner', op='within')
        gdf_wellbuffer = gdf_wellbuffer[['well_id', 'lat', 'lon', 'geometry']]

        # Export the buffer to a shapefile
        # gdf_wellbuffer.to_pickle(config.data_processed / f'Well_buffer_shape' / f"{well_src}_buffers_{(rad_buffer)}mile.pkl")
        gdf_wellbuffer.to_file(config.data_processed / f'Well_buffer_shape' / f"{well_src}_buffers_{(rad_buffer)}mile.shp", driver='ESRI Shapefile')


    #============================ AEM values in the well buffer ===============================

    def get_aem_mean_in_well_buffer(gdfres, wqnodes_2m_gpd, aem_value_type):
        """
        Get the mean resistivity of AEM data in the buffer around each well.

        Parameters:
        - gdfres (gpd.GeoDataFrame): AEM data
        - wqnodes_2m_gpd (gpd.GeoDataFrame): Buffer around each well
        """
        # Make a copy of the AEM data
        aemdata = gdfres.copy()
        
        # Get the intersections between the AEM data and the well buffer
        aem_wq_buff = gpd.overlay(aemdata, wqnodes_2m_gpd, how='intersection')
        
        # Compute the mean resistivity for each well
        aem_wq_buff_aemmean = aem_wq_buff.groupby("well_id").Resistivity.apply(lambda x: np.nanmean(x)).reset_index(name='Resistivity')
        
        # Merge the mean resistivity data with the well buffer data
        aem_wq_buff_aemmean = aem_wq_buff_aemmean.merge(wqnodes_2m_gpd, on='well_id', how='left')
        
        # Select few columns
        aem_wq_buff_aemmean = aem_wq_buff_aemmean[['well_id', 'Resistivity', 'lat', 'lon', 'geometry']]

        if aem_value_type == 'conductivity':
            aem_wq_buff_aemmean = aem_wq_buff_aemmean.rename(columns={'Resistivity': f'thickness_abovCond_{round(cond_thresh*100)}'}) 

        return aem_wq_buff_aemmean

    warnings.filterwarnings("ignore")

    # Get AEM values inside buffer
    aem_inside_buffer = get_aem_mean_in_well_buffer(gdfres= gdfaem, wqnodes_2m_gpd = gdf_wellbuffer,aem_value_type = aem_value_type)

    # Drop geometry column
    aem_inside_buffer2 = aem_inside_buffer.drop(['geometry'], axis=1)

    # Export CSV with AEM values
    if gama_old_new == 1:
        aem_inside_buffer2.to_csv(config.data_processed / f"aem_values/Thickness_abovThresh_{aem_src}_wellsrc_{well_src}_rad_{rad_buffer}mile_lyrs_{aem_lyr_lim}_condThresh_{round(cond_thresh*100)}.csv")
    if gama_old_new == 2:
        aem_inside_buffer2.to_csv(config.data_processed / f"aem_values/Thickness_abovThresh_{aem_src}_wellsrc_{well_src}latest_rad_{rad_buffer}mile_lyrs_{aem_lyr_lim}_condThresh_{round(cond_thresh*100)}.csv")
#%%
well_rad_all = [0.5,1,2,3,4,5]
cond_thresh_values = [0.05, 0.057, 0.066, 0.08, 0.1, 0.15]
aem_lyr_lim_values = [4, 6, 9]
# well_src_values = ['UCD', 'GAMA']
well_src_values = ['GAMA']

for well_src in well_src_values:
    for well_rad_sel in well_rad_all:
        for cond_thresh in cond_thresh_values:
            for aem_lyr_lim in aem_lyr_lim_values:
                get_thickness_well_buffer(well_src=well_src, cond_thresh=cond_thresh, aem_lyr_lim=aem_lyr_lim,rad_buffer = well_rad_sel, gama_old_new=2)

#%%
#============================ Exporting data to kml ==================================
# fiona.supported_drivers['KML'] = 'rw'

# gdf_wellbuffer2 = gdf_wellbuffer.copy()
# gdf_wellbuffer2 = gdf_wellbuffer2[['well_id', 'geometry']]

# # Define the output file name
# gdf_wellbuffer2.to_file(config.data_processed / f'Well_buffer_shape' / f'{well_src}_buffer_{rad_buffer}mile.kml', driver='KML')

# %%

