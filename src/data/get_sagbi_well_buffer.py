#========================================================================================
# This script calculates the average SAGBI within well buffers.
# Steps:
# 1. Create well buffer of given radius
# 2. Average the SAGBI in the well buffers
# 3. Export data to CSV
#========================================================================================
#%%
import sys
sys.path.insert(0,'src')
import config
import pandas as pd
import ppfun as dp
import geopandas as gpd

# file location
file_polut = config.data_raw / "nitrate_data/UCDNitrateData.csv"
file_aem = config.data_processed / 'AEM'

# Read region boundary
cv = dp.get_region(config.shapefile_dir   / "cv.shp")

# sagbi_unmod = dp.get_region(config.data_raw / 'SAGBI/sagbi_unmod/sagbi_unmod.shp')
sagbi_unmod = gpd.read_file(config.data_raw / 'SAGBI/sagbi_unmod/sagbi_unmod.json')


#==================== User Input requred ==========================================
well_src       = 'UCD'          # options: UCD, GAMA
rad_buffer     = 2              # well buffer radius in miles
buffer_flag    = 0              # flag 1: use existing buffer shapefile; 0: create buffer
#==================================================================================


#=========================== Import water quality data ==========================
if well_src == 'GAMA':
    # read gama excel file
    df = pd.read_excel(config.data_gama / 'TULARE_NO3N.xlsx',engine='openpyxl')
    df.rename(columns = {'GM_WELL_ID':'WELL ID', 'GM_LATITUDE':'APPROXIMATE LATITUDE', 'GM_LONGITUDE':'APPROXIMATE LONGITUDE', 'GM_CHEMICAL_VVL': 'CHEMICAL', 'GM_RESULT': 'RESULT','GM_WELL_CATEGORY':'DATASET_CAT','GM_SAMP_COLLECTION_DATE':'DATE'}, inplace = True)
    df['DATE']= pd.to_datetime(df['DATE'])

if well_src == 'UCD':
    # file location
    file_polut = config.data_raw / "nitrate_data/UCDNitrateData.csv"

    # Read nitrate data
    df = dp.get_polut_df(file_sel = file_polut)

# Group the DataFrame by the 'WELL ID' column and take the first values for other selected columns
df = df.groupby('WELL ID')['APPROXIMATE LATITUDE', 'APPROXIMATE LONGITUDE',].apply(lambda x: x.iloc[0]).reset_index()

#======================== Clip around boundary ========================
# converting point resistivity data to polygon mask
# Creating buffer around wells
gdf_wellbuffer_all = dp.get_well_buffer_shape(df,rad_buffer = rad_buffer) 

# Perform  clip on the well data based on selected boundary
gdf_wellbuffer = gpd.sjoin(gdf_wellbuffer_all, cv, how='inner', op='within')
gdf_wellbuffer = gdf_wellbuffer[['well_id', 'lat', 'lon', 'geometry']]

#============================ AEM values in the well buffer ===============================

def get_sagbi_at_wq_buffer(gdf_wellbuffer = gdf_wellbuffer, sagbi = sagbi_unmod):

    """
    Get water quality buffers with average sagbi rating. combine wq values and sagbi rating
    """
    
    g = gpd.overlay(sagbi,gdf_wellbuffer, how='intersection')

    # area weighted sagbi rating calculate
    g['Area_calc'] =g.apply(lambda row: row.geometry.area,axis=1)
    g['area_sagbi'] = g['Area_calc'] * g['sagbi']

    dff = g.groupby(["well_id"]).Area_calc.sum().reset_index()
    dff2 = g.groupby(["well_id"]).area_sagbi.sum().reset_index()
    dff2['area_wt_sagbi'] = dff2['area_sagbi']/ dff['Area_calc']

    dff2 = dff2.drop(['area_sagbi'], axis=1, errors='ignore')
    
    # merge wq data 
    wqbuff_wt_sagbi = g.merge(dff2, on='well_id', how='left')

    wqbuff_wt_sagbi_2 = wqbuff_wt_sagbi.groupby('well_id')['area_wt_sagbi'].mean()

    return wqbuff_wt_sagbi_2

# Get AEM values inside buffer
wqbuff_wt_sagbi = get_sagbi_at_wq_buffer(gdf_wellbuffer = gdf_wellbuffer, sagbi = sagbi_unmod)

# Drop geometry column
# wqbuff_wt_sagbi = wqbuff_wt_sagbi.drop(['geometry'], axis=1)

# Export CSV with AEM values
wqbuff_wt_sagbi.to_csv(config.data_processed / f"sagbi_values/SAGBI_wellsrc_{well_src}_rad_{rad_buffer}mile.csv")

# %%
