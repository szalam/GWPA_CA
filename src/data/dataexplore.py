#%%
from pickle import TRUE
import ppfun as dp
import pandas as pd
import geopandas as gpd

sys.path.insert(0,'src')
import config

# location of input files
file_gwpa = config.shapefile_dir / "ca_statewide_gwpa/CA_Statewide_GWPAs.shp"
file_cafo = config.shapefile_dir / 'cafo_well/CAFO.shp'
file_well = config.shapefile_dir / 'cafo_well/well_count.shp'
file_domain = config.shapefile_dir / "kw" / "Kaweah_subregion.shp"
file_polut = config.nitrate_data_dir / "UCDNitrateData.csv"
file_casgem_measurement = config.data_path / "CASGEM/measurements.csv"
file_casgem_station = config.data_path / "CASGEM/stations.csv"
file_casgem_perforation = config.data_path / "CASGEM/perforations.csv"
file_aem_ca = config.shapefile_dir / 'AEM_DWR/Survey_area4/flowlines/SA4_Flown_Flight_Lines.shp'

# %%
# Read required dataset
# read nitrate data
c = dp.get_polut_df(file_sel = file_polut, chemical = 'NO3')

# read kaweah region
kw = dp.get_region(file_domain)

#%%
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
#%%
# Clip point shapes with area boundary
cmax_c = dp.point_clip(file_sel = None, pt_dt = cmax, reg = kw)
# read well, gwpa, cafo and aem shapes and clip
well = dp.point_clip(file_well, reg = kw)
gwpa = dp.point_clip(file_gwpa, reg = kw)
cafo = dp.point_clip(file_cafo, reg = kw)
aem = dp.point_clip(file_aem_ca, reg = kw)
#%%
# Plot data
dp.plt_domain(cmax_c, mcl = 100, region = kw, cafo_shp = None, gwpa = gwpa, well = casgem_kw, welltype= 'Residential', polut_factor=.05)
# %%
dp.plt_domain(cmax_c, mcl = 100, region = kw, cafo_shp = cafo, gwpa = gwpa, aem = aem, well = casgem_kw, welltype= 'Irrigation', polut_factor=.05)
# %%
