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


sys.path.insert(0,'src')
import config

# location of input files
file_gwpa = config.shapefile_dir / "ca_statewide_gwpa/CA_Statewide_GWPAs.shp"
file_cafo = config.shapefile_dir / 'cafo_well/CAFO.shp'
file_well = config.shapefile_dir / 'cafo_well/well_count.shp'
file_domain = config.shapefile_dir / "kw" / "Kaweah_subregion.shp"
file_polut = config.nitrate_data_dir / "UCDNitrateData.csv"
file_casgem_measurement = config.data_raw / "CASGEM/measurements.csv"
file_casgem_station = config.data_raw / "CASGEM/stations.csv"
file_casgem_perforation = config.data_raw / "CASGEM/perforations.csv"
file_aem_ca = config.shapefile_dir / 'AEM_DWR/Survey_area4/flowlines/SA4_Flown_Flight_Lines.shp'
file_aem_engp = config.shapefile_dir / 'ENGP/'
file_aem_interpolated = config.data_processed / 'AEM/ENGP'
file_gama = config.data_raw / 'GAMA'

MCL = 0
# Read required dataset
# read nitrate data
c_no3 = dp.get_polut_df(file_sel = file_polut)

# read gama excel file
c_gama = pd.read_excel(file_gama / 'Initial_download/TOP 10 CHEMICALS/TULARE_NO3N.xlsx',engine='openpyxl')
c_gama.rename(columns = {'GM_WELL_ID':'WELL ID', 'GM_LATITUDE':'APPROXIMATE LATITUDE', 'GM_LONGITUDE':'APPROXIMATE LONGITUDE', 'GM_CHEMICAL_VVL': 'CHEMICAL', 'GM_RESULT': 'RESULT','GM_WELL_CATEGORY':'DATASET_CAT','GM_SAMP_COLLECTION_DATE':'DATE'}, inplace = True)
c_gama['DATE']= pd.to_datetime(c_gama['DATE'])

#%%
c = c_no3.copy()
# read kaweah region
kw = dp.get_region(file_domain)
# Read CASGEM stations
casgem_st = pd.read_csv(file_casgem_station)
casgem_st = dp.df_to_gpd(casgem_st, lat_col_name = 'latitude', lon_col_name = 'longitude')

# Separate for kaweah
casgem_kw = dp.point_clip(pt_dt = casgem_st, reg = kw)

#%%
# Read nitrate data with max, mean, median for all stations
cmax = dp.get_polut_stat(c, stat_sel = 'max')
# Convert pd dataframe to geopandas
cmax = dp.df_to_gpd(cmax)
# Clip point shapes with area boundary
cmax_c = dp.point_clip(file_sel = None, pt_dt = cmax, reg = kw)
# read well, gwpa, cafo and aem shapes and clip
well = dp.point_clip(file_well, reg = kw)
gwpa = dp.point_clip(file_gwpa, reg = kw)
cafo = dp.point_clip(file_cafo, reg = kw)
aem = dp.point_clip(file_aem_ca, reg = kw)
# read the AEM files from ENGP group
gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
aem2015 = gpd.read_file(file_aem_engp / '2015KaweahAEM.kml', driver='KML')
aem2018 = gpd.read_file(file_aem_engp / '2018KaweahAEM.kml', driver='KML')
aem2020 = gpd.read_file(file_aem_engp / '2020FoothillsAEM.kml', driver='KML')
aemengp = aem2015.append(aem2018.append(aem2020))
#%%
# Plot data
dp.plt_domain(cmax_c, mcl = MCL, region = kw, cafo_shp = None, gwpa = gwpa, well = casgem_kw, welltype= 'Residential', polut_factor=.05)
# dp.plt_domain(cmax_c, mcl = 10, region = kw, cafo_shp = cafo, gwpa = None, aem = aemengp, well = None, welltype= 'Irrigation', polut_factor=.05)
# dp.plt_domain(cmax_c, mcl = 150, region = kw, cafo_shp = None, gwpa = gwpa, aem = aem, well = None, welltype= 'Irrigation', polut_factor=.05)
# %%
#spatial join of gwpa and cmax. 
# cmax_abov_thr = cmax_c[cmax_c.VALUE > 100]
# gwpa_cmax = gpd.sjoin(gwpa, cmax_abov_thr)
# gwpa_cmax.plot()

# Interpolated AEM data import
aem_interp = np.load(file_aem_interpolated / 'max_resistivity_upto_layer_8.npy')
aem_interp_X = np.load(file_aem_interpolated / 'X_for_resistivity_data.npy')
aem_interp_Y = np.load(file_aem_interpolated / 'Y_for_resistivity_data.npy')

#%%
# setting up the model
apmod_l8_c10 = aempol(cmax_c = cmax_c,c_threshold = MCL,gwpa = gwpa, aemlyrlim = 8, file_aem_interpolated = file_aem_interpolated, resistivity_threshold = 20, coord = None,c_above_thres = 1)
apmod_l5_c10 = aempol(cmax_c = cmax_c,c_threshold = MCL,gwpa = gwpa, aemlyrlim = 5, file_aem_interpolated = file_aem_interpolated, resistivity_threshold = 20, coord = None,c_above_thres = 1)
apmod_l12_c10 = aempol(cmax_c = cmax_c,c_threshold = MCL,gwpa = gwpa, aemlyrlim = 12, file_aem_interpolated = file_aem_interpolated, resistivity_threshold = 20, coord = None,c_above_thres = 1)

modelsel = apmod_l8_c10
#%%
# Scatter plot of aem and pollutant concentration
modelsel.get_scatter_aem_polut(gwpaflag = 'All',yaxis_lim = 100)
# Resistivity plot above threshold with the gwpa map
modelsel.get_resistivity_plot_abovthresh()
# AEM resistivity plot inside and outside gwpa
modelsel.get_aem_hist_inout_gwpa()
# %%
modelsel.aem_cmax_gwpa_out.Resistivity.describe()
# %%
