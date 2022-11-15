#%%
import pathlib as plib
import utils

# -- global module filepaths -- #
data_path = utils.get_datafpath()

data_raw = data_path / "raw"
data_processed = data_path / "processed"

nitrate_data_dir = data_raw / "nitrate_data"
shapefile_dir = data_raw / "shapefile"

#%%

crs_albers='epsg:3488' # California Albers
crs_latlon=4326
crs_albers_num = 3488