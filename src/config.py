#%%
import pathlib as plib
import utils

# -- global module filepaths -- #
data_path = utils.get_datafpath()

data_raw = data_path / "raw"
data_processed = data_path / "processed"

data_gama = data_raw / 'GAMA/Initial_download/TOP 10 CHEMICALS'
data_aem_interp = data_processed / 'AEM'
shapefile_dir = data_raw / "shapefile"

#%%

crs_albers='epsg:3488' # California Albers
crs_latlon=4326
crs_albers_num = 3488