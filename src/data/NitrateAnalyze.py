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

import config

sys.path.insert(0,'src')

# location of input files
file_polut = config.data_raw / "nitrate_data/UCDNitrateData.csv"
file_aem_ca = config.shapefile_dir / 'AEM_DWR/Survey_area4/flowlines/SA4_Flown_Flight_Lines.shp'
file_aem = config.data_processed / 'AEM'

#==================== User Input requred ==========================================
MCL = 10
aem_src        = 'DWR'          # options: DWR, ENVGP
aem_reg        = 5              # options: 5, 4. Only applicable when aem_src = DWR
aem_reg2       = 4              # use only if two regions are worked with
aem_lyr_lim    = 9              # options: 9, 8. For DWR use 9,3,20, for ENVGP use 8
aem_value_type = 'conductivity' # options: resistivity, conductivity
aem_stat       = 'mean'         # options: mean, min
reginput       = 'CV'           # options: CV, KW
aem_threshold  = 0             # optional. This is used to plot aem resistivity values> aem_threshold
flag_thresh_c  = 1              # options: 1 = take c above threshold, 0 = don't take
#==================================================================================


#==================== Read required dataset =======================================
# read nitrate data
c_no3 = dp.get_polut_df(file_sel = file_polut)

# read gama excel file
c_gama = pd.read_excel(config.data_gama / 'TULARE_NO3N.xlsx',engine='openpyxl')
c_gama.rename(columns = {'GM_WELL_ID':'WELL ID', 'GM_LATITUDE':'APPROXIMATE LATITUDE', 'GM_LONGITUDE':'APPROXIMATE LONGITUDE', 'GM_CHEMICAL_VVL': 'CHEMICAL', 'GM_RESULT': 'RESULT','GM_WELL_CATEGORY':'DATASET_CAT','GM_SAMP_COLLECTION_DATE':'DATE'}, inplace = True)
c_gama['DATE']= pd.to_datetime(c_gama['DATE'])

c = c_no3.copy()            # options: c_no3.copy(), c_gama.copy()
# read kaweah region
kw = dp.get_region(config.shapefile_dir   / "kw/Kaweah_subregion.shp")
cv = dp.get_region(config.shapefile_dir   / "cv.shp")

# Read CASGEM stations
casgem_st = pd.read_csv(config.data_raw / "CASGEM/stations.csv")
casgem_st = dp.df_to_gpd(casgem_st, lat_col_name = 'latitude', lon_col_name = 'longitude')

# read well, gwpa, cafo and aem shapes and clip
well = dp.point_clip((config.shapefile_dir   / 'cafo_well/well_count.shp'), reg = cv)
gwpa = dp.point_clip((config.shapefile_dir   / "ca_statewide_gwpa/CA_Statewide_GWPAs.shp"), reg = cv)
cafo = dp.point_clip((config.shapefile_dir   / 'cafo_well/CAFO.shp') , reg = cv)
aem = dp.point_clip(file_aem_ca, reg = cv)

# read the AEM files from ENGP group
gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
aem2015 = gpd.read_file(config.shapefile_dir   / 'ENGP/2015KaweahAEM.kml', driver='KML')
aem2018 = gpd.read_file(config.shapefile_dir   / 'ENGP/2018KaweahAEM.kml', driver='KML')
aem2020 = gpd.read_file(config.shapefile_dir   / 'ENGP/2020FoothillsAEM.kml', driver='KML')
aemengp = aem2015.append(aem2018.append(aem2020))

# Groundwater depth data read
gwdp = np.load(config.data_processed / 'GWdepth_interpolated' / 'GWdepth_spring.npy')
gwdp_x = np.load(config.data_processed / 'GWdepth_interpolated' / 'X_cv_gwdepth.npy')
gwdp_y = np.load(config.data_processed / 'GWdepth_interpolated' / 'Y_cv_gwdepth.npy')

gwd_gdf = dp.gwdep_array_gdf(gwdp,gwdp_x,gwdp_y)

# CAFO dating data 
cafo_dts = pd.read_csv(config.data_processed / "CAFO_dating/CA_final_with_dates.csv")
cafo_dts_gdf = gpd.GeoDataFrame(
    cafo_dts, geometry=gpd.points_from_xy(cafo_dts.longitude, cafo_dts.latitude))

#%%
# read sagbi data. Previously exported to json. If first time, then import shp.
# sagbi_unmod = dp.get_region(config.data_raw / 'SAGBI/sagbi_unmod/sagbi_unmod.shp')

# export sagbi if not done before
# sagbi_unmod.to_file(config.data_raw / 'SAGBI/sagbi_unmod/sagbi_unmod.json', driver="GeoJSON")
# sagbi_unmod.to_file(config.data_raw / 'SAGBI/sagbi_unmod/sagbi_unmod.gpkg', driver="GPKG")

sagbi_unmod = gpd.read_file(config.data_raw / 'SAGBI/sagbi_unmod/sagbi_unmod.json')
#%%
#==================================================================================
if reginput == 'CV':
    regsel = cv
if reginput == 'KW':
    regsel = kw
# Separate for kaweah
casgem_kw = dp.point_clip(pt_dt = casgem_st, reg = regsel)

#==================== Nitrate data process =======================================
# Read nitrate data with max, mean, median for all stations
cmax = dp.get_polut_stat(c, stat_sel = 'mean')
# Convert pd dataframe to geopandas
cmax = dp.df_to_gpd(cmax)
# Clip point shapes with area boundary
cmax_c = dp.point_clip(file_sel = None, pt_dt = cmax, reg = None)

#%%
# Plot data
# dp.plt_domain(cmax_c, mcl = MCL, region = kw, cafo_shp = None, gwpa = gwpa, well = None, welltype= 'Residential', polut_factor=.05, c_name = 'Nitrate')
dp.plt_domain(cmax_c, mcl = 10, region = cv, cafo_shp = None, gwpa = None, 
                aem = None, well = None, welltype= None, polut_factor=.03)

#%%
cmax_c.plot(markersize = 1, column= 'WELL ID') # harter data plot

#%%
# ==================== Calling the model =======================================
def aem_dwr_info(file_aem, aem_src, aem_value_type, aem_lyr_lim, aem_reg):
    if aem_src == 'DWR':
        if aem_value_type == 'conductivity':
            aem_fil_loc = file_aem / aem_src / aem_value_type
            interpolated_aem_file = f'{aem_value_type}_lyrs_{aem_lyr_lim}_reg{aem_reg}.npy'
        if aem_value_type == 'resistivity':
                aem_fil_loc = file_aem / aem_src / aem_value_type / aem_stat
                interpolated_aem_file = f'{aem_value_type}_lyrs_{aem_lyr_lim}_reg{aem_reg}.npy'
    return aem_fil_loc, interpolated_aem_file

def aem_envgp_info(file_aem, aem_src, aem_value_type, aem_lyr_lim, aem_reg):
    if aem_src == 'ENVGP':
        if aem_value_type == 'conductivity':
            aem_fil_loc = file_aem / aem_src / aem_value_type
            interpolated_aem_file = f'{aem_value_type}_lyrs_{aem_lyr_lim}.npy'
        if aem_value_type == 'resistivity':
                aem_fil_loc = file_aem / aem_src / aem_value_type / aem_stat
                interpolated_aem_file = f'{aem_value_type}_lyrs_{aem_lyr_lim}.npy'
    return aem_fil_loc, interpolated_aem_file

aem_fil_loc1, interpolated_aem_file1 = aem_dwr_info(file_aem = file_aem, aem_src = aem_src, 
                                                    aem_value_type = aem_value_type, 
                                                    aem_lyr_lim = aem_lyr_lim, aem_reg = aem_reg)
aem_fil_loc2, interpolated_aem_file2 = aem_dwr_info(file_aem = file_aem, aem_src = aem_src, 
                                                    aem_value_type = aem_value_type, 
                                                    aem_lyr_lim = aem_lyr_lim, aem_reg = aem_reg2)

apmod = aempol(cmax_c = cmax_c,c_threshold = MCL,gwpa = gwpa, 
                            aemlyrlim = aem_lyr_lim, file_loc_interpolated = aem_fil_loc1, 
                            file_aem_interpolated = interpolated_aem_file1,
                            resistivity_threshold = aem_threshold, coord = None,
                            c_above_thres = flag_thresh_c,
                            aemsrc = aem_src,aemregion = aem_reg,
                            aem_value_type = aem_value_type,
                            gwd_gdf = gwd_gdf,
                            sagbi = sagbi_unmod)

apmod2 = aempol(cmax_c = cmax_c,c_threshold = MCL,gwpa = gwpa, 
                            aemlyrlim = aem_lyr_lim, file_loc_interpolated = aem_fil_loc2, 
                            file_aem_interpolated = interpolated_aem_file2,
                            resistivity_threshold = aem_threshold, coord = None,
                            c_above_thres = flag_thresh_c,
                            aemsrc = aem_src,aemregion = aem_reg2,
                            aem_value_type = aem_value_type,
                            gwd_gdf = gwd_gdf,
                            sagbi = sagbi_unmod)

modelsel = apmod
#%%
apmod.get_scatter_aem_polut(gwpaflag = 'All',datareg2 = apmod2, yaxis_lim = 1000) # datareg2 = None with plot only one region
#%%
# apmod.get_sagbi_at_wqnodes()

#%%
# # Exporting water quality data to shapefile to be used for other purpose (such as land use extraction using GEE)
# apmodcombine = pd.concat([apmod.get_gwdepth_at_wqnodes(), apmod2.get_gwdepth_at_wqnodes()])
# apmodcombine_sel = apmodcombine[['WELL ID', 'geometry','gwdep','Resistivity', 'VALUE']].copy()
# apmodcombine_sel.to_file(config.data_processed / 'NO3_loc_abovthreshold/NO3_abovthreshold.shp')  

#%%
apmod.get_aem_conductivity_plot(reg = cv, conductivity_max_lim = .18)
apmod2.get_aem_conductivity_plot(reg = cv, conductivity_max_lim = .18)

#%%
# apmod.get_aem_conductivity_plot_tworegions(apmod2, reg=cv,conductivity_max_lim = .035,
#                                         vmax_in= .11) #0.18

#%%
dp.plt_domain(cmax_c, mcl = MCL, region = modelsel.get_aem_mask(), cafo_shp = None, 
            gwpa = None, aem = None, well = None, welltype= None, polut_factor=.05)

#%%
apmod.get_aem_riskscore_plot(reg = apmod.get_aem_mask(),risk_interval =.1)

#%%
apmod2.get_aem_values().Resistivity.describe()
pd.concat([apmod.get_aem_values().Resistivity, apmod2.get_aem_values().Resistivity]).describe()
#%%
# t-test to check if data dat are statisticall different
t_check=stats.ttest_ind(modelsel.aem_cmax_gwpa_out.VALUE,modelsel.aem_cmax_gwpa_in.VALUE)
t_check
alpha=0.05
if(t_check[1]<alpha):
    print('A different from B')

#%%
# Resistivity plot above threshold with the gwpa map
apmod.get_aemzone_plot_abovthresh(0.1)
#%%
# AEM resistivity plot inside and outside gwpa
apmod.get_aem_hist_inout_gwpa(xlim_sel = .4)
apmod.aem_cmax_gwpa_out.Resistivity.describe()
cmax_c.describe()
# %%
# separate nitrate data with concentration above 10 mg/L, then get statistics of the AEM values
cmax_c_mclup = modelsel.aem_cmax_gwpa_out[modelsel.aem_cmax_gwpa_out['VALUE']>=10]
cmax_c_mclup.Resistivity.describe()
# %%

#=====================================================================================
#======== Scatter plot of nitrate and other (sagbi, water depth, aem) at wq node ===== 
#=====================================================================================

# Scatter plot aem vs nitrate, with color of groundwater depth
apmod.get_scatter_aem_polut_w_gwdepth(yaxis_lim = 500, xaxis_lim = .3,p_size = .2)

# Scatter plot aem vs nitrate, with color of sagbi values at wq nodes
apmod.get_scatter_aem_polut_w_sagbi(datareg2 = apmod2, sagbi_var = 'sagbi',
                                    sagbi_rating_min = 0, sagbi_rating_max = 100,
                                    yaxis_lim= 400,xaxis_lim=.3)
#%%
# Scatter plot of aem vs sagbi at wq node
apmod.get_scatter_aem_vs_sagbi(datareg2 = apmod2, sagbi_var = 'sagbi')

#=====================================================================================
#====================== Well buffer create and extract data at buffers================
#=====================================================================================
#%%
# sagbi vs polut concentration
apmod.get_scatter_sagbi_polut_in_buffer(ylim_sel = 1000)

#%%
# aem vs sagbi
apmod.get_scatter_sagbi_aem_in_buffer(datareg2 = apmod2)

#%%
# aem vs polut concentration
apmod.get_scatter_aem_polut_in_bufferzone(datareg2 = apmod2, yaxis_lim = None, 
                                            xaxis_lim = None)


#=====================================================================================
#================================= CAFO dating data ==================================
#=====================================================================================
#%%
# plot cafo locations
dp.plt_domain(cafo_dts_gdf, mcl = None , region = cv, cafo_shp = None, gwpa = None, aem = None, well = None, welltype = None, polut_factor = None, c_name = 'Cafos dated')


#%%
aem_wq_buff_aemmean_gdf = gpd.GeoDataFrame(apmod.aem_wq_buff_aemmean, geometry='geometry')
aem_wq_buff_aemmean_gdf2 = gpd.overlay(cafo_dts_gdf,aem_wq_buff_aemmean_gdf, how='intersection')
aem_wq_buff_aemmean_cafopopsum_gdf2 = aem_wq_buff_aemmean_gdf2.groupby(["WELL ID"]).Cafo_Population.sum().reset_index()
aem_wq_buff_aemmean_cafopopsum_gdf2 = aem_wq_buff_aemmean_cafopopsum_gdf2.merge(apmod.aem_wq_buff_aemmean, on='WELL ID', how='left')


#%%
fig, ax = plt.subplots(figsize=(7, 7))
plt.scatter(aem_wq_buff_aemmean_cafopopsum_gdf2['Cafo_Population'],aem_wq_buff_aemmean_cafopopsum_gdf2['VALUE'],
            color = 'red', s = 1)
# plt.ylim([0,1000])
# plt.xlim([0,0.9])
plt.xlabel('CAFO Polulation', fontsize=20)
plt.ylabel('Nitrate concentration [mg/l]', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=17)
plt.show()

#%%
fig, ax = plt.subplots(figsize=(7, 7))
plt.scatter(aem_wq_buff_aemmean_cafopopsum_gdf2['Resistivity'],aem_wq_buff_aemmean_cafopopsum_gdf2['Cafo_Population'],
            color = 'red', s = 1)
# plt.ylim([0,1000])
# plt.xlim([0,0.9])
plt.xlabel('Conductivity', fontsize=20)
plt.ylabel('CAFO Polulation', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=17)
plt.show()






#=============== All other useful plots ==============================
#%%
dp.get_polut_scatter(cmax_c,x_val = 'GM_TOP_DEPTH_OF_SCREEN_FT',yaxis_lim= 50)
dp.get_polut_scatter(cmax_c,x_val = 'GM_WELL_DEPTH_FT',yaxis_lim= 50)

#%%
modelsel.get_scatter_aem_polut_w_wellproperty(yaxis_lim = 50, wellproperty = 'GM_WELL_DEPTH_FT',p_size = 15.3) #GM_WELL_DEPTH_FT, GM_TOP_DEPTH_OF_SCREEN_FT
# plt.scatter(cmax_c.GM_WELL_DEPTH_FT, cmax_c.VALUE)

#%%
#Summary of of AEM values. If I separate everything above 19 as vulnerable, it removes 50% of data
# apmod_l8_c10_resmin.get_aem_values().Resistivity.describe()
# %%
#spatial join of gwpa and cmax. 
# cmax_abov_thr = cmax_c[cmax_c.VALUE > 100]
# gwpa_cmax = gpd.sjoin(gwpa, cmax_abov_thr)
# gwpa_cmax.plot()

# Interpolated AEM data import
# aem_interp = np.load(file_aem_interpolated / 'min_resistivity_upto_layer_8.npy')
# aem_interp_X = np.load(file_aem_interpolated / 'X_for_resistivity_data.npy')
# aem_interp_Y = np.load(file_aem_interpolated / 'Y_for_resistivity_data.npy')
