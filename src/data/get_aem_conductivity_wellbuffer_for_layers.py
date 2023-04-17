#========================================================================================
# This script calculates the depth weighted conductivity within well buffers. 
# Depth weighted conductivity = (1/resistivity)*layer depth
# Steps:
# 1. Create well buffer of given radius
# 2. Average the AEM conductivity or resistivity in the well buffers
# 3. Export data to CSV
# 4. Loop the above over layer depths
#========================================================================================
#%%
import sys
sys.path.insert(0,'src')
import config
import warnings
import fiona
import pandas as pd
import ppfun as dp
import numpy as np
import geopandas as gpd
from tqdm import tqdm
from contextlib import redirect_stdout

# file location
file_aem = config.data_processed / 'AEM'

#==================== User Input requred ==========================================

#==================== User Input requred ==========================================
aem_src        = 'DWR'          # options: DWR, ENVGP
well_src       = 'UCD'          # options: UCD, GAMA
aem_region = [4, 5, 6, 7] 
aem_value_type = 'conductivity' # options: resistivity, conductivity
aem_stat       = 'mean'         # options: mean, min
rad_buffer     = 1              # well buffer radius in miles
buffer_flag    = 0              # flag 1: use existing buffer shapefile; 0: create buffer
#==================================================================================


#=========================== Import water quality data ==========================
if well_src == 'GAMA':
    # read gama excel file
    # df = pd.read_excel(config.data_gama / 'TULARE_NO3N.xlsx',engine='openpyxl')
    # df.rename(columns = {'GM_WELL_ID':'WELL ID', 'GM_LATITUDE':'APPROXIMATE LATITUDE', 'GM_LONGITUDE':'APPROXIMATE LONGITUDE', 'GM_CHEMICAL_VVL': 'CHEMICAL', 'GM_RESULT': 'RESULT','GM_WELL_CATEGORY':'DATASET_CAT','GM_SAMP_COLLECTION_DATE':'DATE'}, inplace = True)
    # df['DATE']= pd.to_datetime(df['DATE'])
    file_polut = config.data_gama_all / 'CENTRALVALLEY_NO3N_GAMA.csv'
    df = dp.get_polut_df(file_sel = file_polut)
    df.rename(columns = {'GM_WELL_ID':'WELL ID', 'GM_LATITUDE':'APPROXIMATE LATITUDE', 'GM_LONGITUDE':'APPROXIMATE LONGITUDE', 'GM_CHEMICAL_VVL': 'CHEMICAL', 'GM_RESULT': 'RESULT','GM_WELL_CATEGORY':'well_type','GM_SAMP_COLLECTION_DATE':'DATE'}, inplace = True)
    df['DATE']= pd.to_datetime(df['DATE'])

if well_src == 'UCD':
    # file location
    file_polut = config.data_raw / "nitrate_data/UCDNitrateData.csv"

    # Read nitrate data
    df = dp.get_polut_df(file_sel = file_polut)

# Group the DataFrame by the 'WELL ID' column and take the first values for other selected columns
df = df.groupby('WELL ID')['APPROXIMATE LATITUDE', 'APPROXIMATE LONGITUDE',].apply(lambda x: x.iloc[0]).reset_index()

#============================ Import AEM data ===============================

def aem_info(file_aem, aem_src, aem_value_type, aem_reg, lyr_num):
    aem_fil_loc = file_aem / aem_src / aem_value_type / 'conductivity_each_layer'
    if aem_src == 'DWR':
        interpolated_aem_file = f'{aem_value_type}_lyr_{lyr_num}_reg{aem_reg}.npy'
    elif aem_src == 'ENVGP':
        interpolated_aem_file = f'{aem_value_type}_lyr_{lyr_num}.npy'
    return aem_fil_loc, interpolated_aem_file
lyr_num = 0
for i in tqdm(range(20)):
    aem_args = []
    lyr_num = lyr_num + 1
    for lseq, aem_reg in enumerate(aem_region):
        # call the function
        aem_fil_loc, interpolated_aem_file = aem_info(file_aem, aem_src, aem_value_type, aem_reg,lyr_num)
        aem_args.append({
            'file_loc_interpolated': aem_fil_loc, 
            'file_aem_interpolated': interpolated_aem_file, 
            'aemregion': aem_reg, 
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

    if 'gdf_wellbuffer' not in locals():
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
            aem_wq_buff_aemmean = aem_wq_buff_aemmean.rename(columns={'Resistivity': 'Conductivity'}) 

        return aem_wq_buff_aemmean

    warnings.filterwarnings("ignore")

    # Get AEM values inside buffer
    aem_inside_buffer = get_aem_mean_in_well_buffer(gdfres= gdfaem, wqnodes_2m_gpd = gdf_wellbuffer,aem_value_type = aem_value_type)


    if lyr_num == 1:
        aem_inside_buffer.rename(columns={'Conductivity':f'Conductivity_depthwtd_lyr{lyr_num}'}, inplace=True)
        # Drop geometry column
        aem_inside_buffer2 = aem_inside_buffer.drop(['geometry'], axis=1)
        aemvalue_all_layer = aem_inside_buffer2.copy()
        
    if lyr_num != 1:
        # Drop geometry column
        aem_inside_buffer2 = aem_inside_buffer.drop(['lat','lon','geometry'], axis=1)
        aem_inside_buffer2.rename(columns={'Conductivity':f'Conductivity_depthwtd_lyr{lyr_num}_rad_{rad_buffer}mile'}, inplace=True)
        aemvalue_all_layer = aemvalue_all_layer.merge(aem_inside_buffer2, on='well_id')

# Export CSV with AEM values
aemvalue_all_layer.to_csv(config.data_processed / f"aem_values/AEMsrc_{aem_src}_wellsrc_{well_src}_rad_{rad_buffer}mile_layerwide.csv")

# %%

