import sys
import config
import ppfun as dp
import pandas as pd
import numpy as np
import geopandas as gpd

sys.path.insert(0,'src')

# location of input files
file_polut = config.data_raw / "nitrate_data/UCDNitrateData.csv"
file_aem_ca = config.shapefile_dir / 'AEM_DWR/Survey_area4/flowlines/SA4_Flown_Flight_Lines.shp'
file_aem = config.data_processed / 'AEM'

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
cafo_dts_gdf = gpd.GeoDataFrame(cafo_dts, geometry=gpd.points_from_xy(cafo_dts.longitude, cafo_dts.latitude))

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