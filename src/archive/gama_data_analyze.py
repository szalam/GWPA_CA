#%%
import sys
from pickle import TRUE
import ppfun as dp
import numpy as np
import pandas as pd
import geopandas as gpd
import fiona
import warnings
from shapely.ops import nearest_points, Point
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from aempolutmod import aempol
from matplotlib.colors import LogNorm
from scipy import stats

sys.path.insert(0,'src')
import config

# location of input files
file_shapefile = config.shapefile_dir  
file_polut = config.nitrate_data_dir / "UCDNitrateData.csv"
file_aem_ca = config.shapefile_dir / 'AEM_DWR/Survey_area4/flowlines/SA4_Flown_Flight_Lines.shp'
file_aem_interpolated = config.data_processed / 'AEM/ENGP'
file_gama = config.data_raw / 'GAMA'

c_name = 'TULARE_TCPR123.xlsx' #'TULARE_PCE.xlsx' 'TULARE_TCE.xlsx' #'TULARE_TCPR123.xlsx'

MCL = 0
# Read required dataset
# read gama excel file
c_gama = pd.read_excel(file_gama / 'Initial_download/TOP 10 CHEMICALS' / c_name,engine='openpyxl')
c_gama.rename(columns = {'GM_WELL_ID':'WELL ID', 'GM_LATITUDE':'APPROXIMATE LATITUDE', 'GM_LONGITUDE':'APPROXIMATE LONGITUDE', 'GM_CHEMICAL_VVL': 'CHEMICAL', 'GM_RESULT': 'RESULT','GM_WELL_CATEGORY':'DATASET_CAT','GM_SAMP_COLLECTION_DATE':'DATE'}, inplace = True)
c_gama['DATE']= pd.to_datetime(c_gama['DATE'])

#%%
c = c_gama.copy() #c_no3.copy(), c_gama
# read kaweah region
kw = dp.get_region(file_shapefile / "kw/Kaweah_subregion.shp")
cv = dp.get_region(file_shapefile / "cv.shp")

# Read CASGEM stations
casgem_st = pd.read_csv(config.data_raw / "CASGEM/stations.csv")
casgem_st = dp.df_to_gpd(casgem_st, lat_col_name = 'latitude', lon_col_name = 'longitude')

# Separate for kaweah
casgem_kw = dp.point_clip(pt_dt = casgem_st, reg = kw)

#%%
# Read nitrate data with max, mean, median for all stations
cmax = dp.get_polut_stat(c, stat_sel = 'mean')
# Convert pd dataframe to geopandas
cmax = dp.df_to_gpd(cmax)
# Clip point shapes with area boundary
cmax_c = dp.point_clip(file_sel = None, pt_dt = cmax, reg = None)
# read well, gwpa, cafo and aem shapes and clip
well = dp.point_clip((file_shapefile / 'cafo_well/well_count.shp'), reg = cv)
gwpa = dp.point_clip((file_shapefile / "ca_statewide_gwpa/CA_Statewide_GWPAs.shp"), reg = cv)
cafo = dp.point_clip((file_shapefile / 'cafo_well/CAFO.shp') , reg = cv)
aem = dp.point_clip(file_aem_ca, reg = cv)
# read the AEM files from ENGP group
gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
aem2015 = gpd.read_file(file_shapefile / 'ENGP/2015KaweahAEM.kml', driver='KML')
aem2018 = gpd.read_file(file_shapefile / 'ENGP/2018KaweahAEM.kml', driver='KML')
aem2020 = gpd.read_file(file_shapefile / 'ENGP/2020FoothillsAEM.kml', driver='KML')
aemengp = aem2015.append(aem2018.append(aem2020))
#%%
# Plot data
# dp.plt_domain(cmax_c, mcl = MCL, region = kw, cafo_shp = None, gwpa = gwpa, well = None, welltype= 'Residential', polut_factor=.05, c_name = 'Nitrate')
dp.plt_domain(cmax_c, mcl = MCL, region = cv, cafo_shp = None, gwpa = gwpa, aem = None, well = None, welltype= None, polut_factor=.05)
#%%
cmax_c.plot(markersize = 1, column= 'WELL ID') # harter data plot
#%%
dp.get_polut_scatter(cmax_c,x_val = 'GM_TOP_DEPTH_OF_SCREEN_FT',yaxis_lim= 2)
dp.get_polut_scatter(cmax_c,x_val = 'GM_WELL_DEPTH_FT',yaxis_lim= 2)

# plt.scatter(cmax_c.GM_WELL_DEPTH_FT, cmax_c.VALUE)

# %%
#spatial join of gwpa and cmax. 
# cmax_abov_thr = cmax_c[cmax_c.VALUE > 100]
# gwpa_cmax = gpd.sjoin(gwpa, cmax_abov_thr)
# gwpa_cmax.plot()

# Interpolated AEM data import
# aem_interp = np.load(file_aem_interpolated / 'min_resistivity_upto_layer_8.npy')
# aem_interp_X = np.load(file_aem_interpolated / 'X_for_resistivity_data.npy')
# aem_interp_Y = np.load(file_aem_interpolated / 'Y_for_resistivity_data.npy')

#%%
# setting up the model
apmod_l8_c10_resmean = aempol(cmax_c = cmax_c,c_threshold = MCL,gwpa = gwpa, aemlyrlim = 8, file_aem_interpolated = file_aem_interpolated, resistivity_threshold = 20, coord = None,c_above_thres = 1,aemstat = 'mean',aemsrc = "DWR")
apmod_l8_c10_resmin = aempol(cmax_c = cmax_c,c_threshold = MCL,gwpa = gwpa, aemlyrlim = 8, file_aem_interpolated = file_aem_interpolated, resistivity_threshold = 20, coord = None,c_above_thres = 1,aemstat = 'min',aemsrc =  'DWR')

# apmod_l5_c10 = aempol(cmax_c = cmax_c,c_threshold = MCL,gwpa = gwpa, aemlyrlim = 5, file_aem_interpolated = file_aem_interpolated, resistivity_threshold = 20, coord = None,c_above_thres = 1,aemstat = 'mean')
# apmod_l12_c10 = aempol(cmax_c = cmax_c,c_threshold = MCL,gwpa = gwpa, aemlyrlim = 12, file_aem_interpolated = file_aem_interpolated, resistivity_threshold = 20, coord = None,c_above_thres = 1, aemstat = 'mean')

#%%
modelsel = apmod_l8_c10_resmin
#%%
apmod_l8_c10_resmin.get_aem_resistivity_plot(kw)
apmod_l8_c10_resmean.get_aem_resistivity_plot(gwpa)

#%%
dp.plt_domain(cmax_c, mcl = MCL, region = modelsel.get_aem_mask(), cafo_shp = None, gwpa = None, aem = None, well = None, welltype= None, polut_factor=.05)
# %%
modelsel.get_aem_riskscore_plot(gwpa,risk_interval =2)
#%%
modelsel.get_aem_riskscore_plot(cv,risk_interval =2)
#%%
#Summary of of AEM values. If I separate everything above 19 as vulnerable, it removes 50% of data
apmod_l8_c10_resmin.get_aem_values().Resistivity.describe()
#%%
apmod_l8_c10_resmean.get_aem_values().Resistivity.describe()

#%%
# Scatter plot of aem and pollutant concentration
modelsel.get_scatter_aem_polut(yaxis_lim = 2, gwpaflag = 'both',YlabelC = '1,2,3 Tricholoropropane',unitylabel = 'UG/L')

#%%
# t-test to check if data dat are statisticall different
t_check=stats.ttest_ind(modelsel.aem_cmax_gwpa_out.VALUE,modelsel.aem_cmax_gwpa_in.VALUE)
t_check
alpha=0.05
if(t_check[1]<alpha):
    print('A different from B')
else:
    print('A and B are not different')

#%%
modelsel.get_scatter_aem_polut_w_wellproperty(yaxis_lim = 2, wellproperty = 'GM_WELL_DEPTH_FT',p_size = 15.3) #GM_WELL_DEPTH_FT, GM_TOP_DEPTH_OF_SCREEN_FT
#%%
# Resistivity plot above threshold with the gwpa map
modelsel.get_aemzone_plot_abovthresh(29.55)
#%%
# AEM resistivity plot inside and outside gwpa
modelsel.get_aem_hist_inout_gwpa(xlim_sel = 150)
# %%
modelsel.aem_cmax_gwpa_out.Resistivity.describe()
# %%
cmax_c.describe()
# %%
# separate nitrate data with concentration above 10 mg/L, then get statistics of the AEM values
cmax_c_mclup = modelsel.aem_cmax_gwpa_out[modelsel.aem_cmax_gwpa_out['VALUE']>=.0005]
cmax_c_mclup.Resistivity.describe()
# %%
