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
from shapely.geometry import Point
from geopandas import GeoDataFrame
from pyproj import CRS
from tqdm import tqdm
from contextlib import redirect_stdout

# file location
file_polut = config.data_raw / "nitrate_data/UCDNitrateData.csv"
file_aem = config.data_processed / 'AEM'

#==================== User Input requred ==========================================
aem_src        = 'DWR'          # options: DWR, ENVGP
well_src       = 'UCD'          # options: UCD, GAMA
aem_reg        = 5              # options: 5, 4. Only applicable when aem_src = DWR
aem_reg2       = 4              # use only if two regions are worked with
aem_lyr_lim    = 9              # options: 9, 8. For DWR use 9,3,20, for ENVGP use 8
aem_value_type = 'conductivity' # options: resistivity, conductivity
aem_stat       = 'mean'         # options: mean, min
rad_buffer     = 2              # well buffer radius in miles
buffer_flag    = 0              # flag 1: use existing buffer shapefile; 0: create buffer
#==================================================================================

#%%
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

#============================ Import AEM data ===============================

def aem_info(file_aem, aem_src, aem_value_type, aem_lyr_lim, aem_reg):
    aem_fil_loc = file_aem / aem_src / aem_value_type
    if aem_src == 'DWR':
        interpolated_aem_file = f'{aem_value_type}_lyrs_{aem_lyr_lim}_reg{aem_reg}.npy'
    elif aem_src == 'ENVGP':
        interpolated_aem_file = f'{aem_value_type}_lyrs_{aem_lyr_lim}.npy'
    return aem_fil_loc, interpolated_aem_file

# call the function
aem_fil_loc1, interpolated_aem_file1 = aem_info(file_aem, aem_src, aem_value_type, aem_lyr_lim, aem_reg)
aem_fil_loc2, interpolated_aem_file2 = aem_info(file_aem, aem_src, aem_value_type, aem_lyr_lim, aem_reg2)


# Create a list of arguments for the get_aem_from_npy function
aem_args = [
    {
        'file_loc_interpolated': aem_fil_loc1, 
        'file_aem_interpolated': interpolated_aem_file1, 
        'aemregion': aem_reg, 
        'aemsrc': aem_src
    },
    {
        'file_loc_interpolated': aem_fil_loc2, 
        'file_aem_interpolated': interpolated_aem_file2, 
        'aemregion': aem_reg2, 
        'aemsrc': aem_src
    }
]

# Use a list comprehension to apply the get_aem_from_npy function to each set of arguments
gdf_aem_list = [dp.get_aem_from_npy(**args) for args in aem_args]

# Concatenate the dataframes into one
gdfaem = pd.concat(gdf_aem_list)

#======================== Get boundary of AEM data ========================
# converting point resistivity data to polygon mask
gdftmp = gdfaem.copy()
gdftmp=gdftmp.dropna(subset=['Resistivity'])
# %%
import matplotlib.pyplot as plt
plt.hist(gdfaem.Resistivity)
gdfaem.Resistivity.describe()
# %%
