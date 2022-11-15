#%%
import sys 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import geopandas as gpd
import wlfun as wl
from arcgis.gis import GIS
import matplotlib.dates as mdates
from pygeostat.data import iotools
from sklearn.linear_model import LinearRegression
from shapely.geometry import Point
import matplotlib
from wlob import wlob
from scipy.spatial import cKDTree
from tqdm import tqdm
import cmcrameri.cm as cmc

sys.path.insert(0,'src')
import config

well_filter = 0
# %%
# Loading relevant shapefiles
# First, we must load some relevant shapefiles. We'll grab California, the Central Valley, the Corcorcan Clay, and the zone map for CVHM (to do: add CV2Sim). I project everything onto Albers as an arbitrary choice, but as long as all crs's for the dataframes align, any geographic projection should be fine.
CA_map = gpd.read_file(config.shapefile_dir / 'CA_State_TIGER2016.shp')
CA_map.to_crs(epsg=config.crs_latlon, inplace=True)

CV_map = gpd.read_file(config.shapefile_dir / 'Alluvial_Bnd.shp')
CV_map.to_crs(epsg=config.crs_latlon, inplace=True)

CC_map = gpd.read_file(config.shapefile_dir / 'corcoran_clay_depth_feet.shp')
CC_map.to_crs(epsg=config.crs_latlon, inplace=True)

#===============================
## Periodic Data
# %%
well_measurements = pd.read_csv(config.data_raw / 'CASGEM/measurements.csv', parse_dates=['msmt_date'])
well_stations = pd.read_csv(config.data_raw /'CASGEM/stations.csv')
well_perforations = pd.read_csv(config.data_raw / 'CASGEM/perforations.csv')

# calling the well object
wlo = wlob(well_stations,well_measurements,well_perforations, CA_map, CV_map, CC_map)

#Read CVHM zone data
zone_map = wlo.get_cvhm_lyr()
zone_map.info()

# %%
#plot study domain
wl.get_cv_map(CA = CA_map,CV = CV_map, corcoran = CC_map, zone_map = None)
# print(well_measurements["wlm_qa_desc"].unique())

# If interested to remove wells with specific comments, such as recent pumping. use following
if well_filter == 1:
    well_stations = wlo.well_filter_qa()
#well_stations.info()

# %%
# combine well perforation, stations, and measurements
well_msp = wlo.well_meas_perfor_combine()
well_msp.head()

#%%
cv_all_sites = wlo.get_well_from_use_region(filter_qa = None, well_use = None, all_wells = 1, CV_map = CV_map, CC_map = None)
cv_obs_sites = wlo.get_well_from_use_region(filter_qa = None, well_use = 'Observation', all_wells = None, CV_map = CV_map, CC_map = None)
cv_ag_sites = wlo.get_well_from_use_region(filter_qa = None, well_use = 'Irrigation', all_wells = None, CV_map = CV_map, CC_map = None)
cv_industry_sites = wlo.get_well_from_use_region(filter_qa = None, well_use = 'Industrial', all_wells = None, CV_map = CV_map, CC_map = None)
cv_resident_sites = wlo.get_well_from_use_region(filter_qa = None, well_use = 'Residential', all_wells = None, CV_map = CV_map, CC_map = None)


#%%
# Process zones and get shallow wells
shallow_wells_all = wlo.get_shallow_wells(zone_map, cv_all_sites,cv_obs_sites, well_type = 'all')
shallow_wells = wlo.get_shallow_wells(zone_map, cv_all_sites,cv_obs_sites, well_type = 'obs')

# thickness files are in meters
shallow_wells["layer1_thk"].describe()
# %%
#plt.figure(figsize=(8,6))


# %%
shallow_measurements = well_msp[well_msp["site_code"].isin(shallow_wells["site_code"])]
shallow_measurements_all = well_msp[well_msp["site_code"].isin(shallow_wells_all["site_code"])]
shallow_measurements.head()

#%%
shallow_measurements_all.to_csv(config.data_processed / 'shallow_wells_all_cvhm.csv')
#%%
st_list = ['351296N1193938W004','361600N1197800W001','362122N1192962W001','362301N1192828W001','362400N1198300W002']
wlo.get_wells_by_aquifer(cv_obs_sites,shallow_wells,st_list)

#%%

# Plot well time series if number of observation is above a threshold
site_codes = list(shallow_measurements["site_code"].unique())
fig = plt.figure(figsize=(16,12))
count = 0
for site_code in site_codes:
    single_well = shallow_measurements[shallow_measurements["site_code"].isin([site_code])]
    single_well = single_well.sort_values(["msmt_date"], ascending=[True])
    if single_well["gse_gwe"].count() > 100:
        plt.plot(single_well["msmt_date"], single_well["gse_gwe"], label=site_codes[count])
        count += 1
    if count == 5:
        break
plt.title('Select Monitoring Wells Time Series', fontsize=20)
plt.ylabel('Depth to Groundwater (ft)', fontsize=16)
plt.setp(plt.gca().get_xticklabels(), rotation=30, horizontalalignment='right')
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
#plt.gca().invert_xaxis()
plt.grid(zorder=0)
plt.legend(fontsize=16)

# %%
wlo.get_well_plot_by_type(cv_sites = cv_resident_sites, well_label = 'Residential')
wlo.get_well_plot_by_type(cv_sites = cv_obs_sites, well_label = 'Observation')
wlo.get_well_plot_by_type(cv_sites = cv_ag_sites, well_label = 'Agricultural')
wlo.get_well_plot_by_type(cv_sites = cv_industry_sites, well_label = 'Industrial')
#%%
wlo.get_well_plot_by_type(cv_sites = cv_all_sites, well_label = 'All')



#----------------------------------
# well interpolation
#----------------------------------

#%%
CA_map = gpd.read_file(config.shapefile_dir / 'CA_State_TIGER2016.shp')
CA_map.to_crs(epsg='4326', inplace=True)

CV_map = gpd.read_file(config.shapefile_dir / 'Alluvial_Bnd.shp')
CV_map.to_crs(epsg='4326', inplace=True)
# %%

# user input
in_path = config.data_processed / 'shallow_wells_all_cvhm.csv'
target_year = '2021'
target_season = 'spring'
monitoring_weight = 0.8
n_realizations = 10000
crs_latlon='epsg:4326' # WGS 84
crs_projected='epsg:3488' # California Albers

x_ = np.arange(-123, -118, 0.10)
y_ = np.arange(34.5, 41, 0.10)
print(len(x_))
print(len(y_))
xx,yy = np.meshgrid(x_,y_,indexing = 'ij')
xx_flat = xx.reshape(-1)
yy_flat = yy.reshape(-1)

meshgrid_points = [Point(xy) for xy in zip(xx_flat, 
                                           yy_flat)]
meshgrid_frame = gpd.GeoDataFrame(meshgrid_points, crs=crs_latlon, geometry=meshgrid_points)
# %%
new_binary = mpl.colors.ListedColormap(np.array([[1., 1., 1., 1.],
       [0., 0., 0., 0.]]))
CV_overlay = np.zeros(xx.shape)
CV_geom = CV_map['geometry'].values[0]
for i in tqdm(range(len(CV_overlay))):
    for j in range(len(CV_overlay[0,:])):
        potentialPoint = Point(xx[i,j], yy[i,j])
        if potentialPoint.within(CV_geom):
            CV_overlay[i,j] = 1
# %%
well_measurements = pd.read_csv(in_path, parse_dates=['msmt_date'])
print(len(well_measurements))
site_points = [Point(xy) for xy in zip(well_measurements["longitude"], well_measurements["latitude"])]


well_measurements = gpd.GeoDataFrame(well_measurements, crs=crs_latlon, geometry=site_points)
well_measurements = well_measurements.to_crs(crs_projected)
# %%
if target_season == 'spring':
    start_month = '01'
    mid_month = '03'
    end_month = '06'
else:
    start_month = '06'
    mid_month = '09'
    end_month='12'

mask = ((well_measurements['msmt_date'] > target_year+start_month+'01') & 
        (well_measurements['msmt_date'] <= target_year+end_month+'01'))
# %%

timeframed_measurements = well_measurements.loc[mask]
timeframed_measurements = timeframed_measurements[timeframed_measurements['gse_gwe'].notnull()]
# %%

timeframed_measurements = timeframed_measurements[timeframed_measurements['gse_gwe'] > 0]
# %%
timeframed_measurements['drop'] = True
# %%
timeframed_measurements_x = timeframed_measurements['geometry'].x
timeframed_measurements_y = timeframed_measurements['geometry'].y
timeframed_measurements_lat = timeframed_measurements["latitude"]
timeframed_measurements_lon = timeframed_measurements["longitude"]
timeframed_measurements_z = timeframed_measurements['gse_gwe']
timeframed_measurements_type = timeframed_measurements["well_use"]
# %%

meshgrid_frame = meshgrid_frame.to_crs(crs_projected)
# %%


xx_flat = meshgrid_frame['geometry'].x
yy_flat = meshgrid_frame['geometry'].y
xx_yy = np.array((xx_flat,yy_flat)).T

# %%
# Stacking the x, y, and z data together in a new array
timeframed_measurements_xyz = np.vstack((timeframed_measurements_x, timeframed_measurements_y, timeframed_measurements_z)).T
# %%
bootstrap_timeframed_measurements_xyz = np.zeros((n_realizations, int(len(timeframed_measurements_xyz)*0.8), 3))
p_array = np.ones((int(len(timeframed_measurements_xyz))))*(1-monitoring_weight)/len(
    timeframed_measurements_type[timeframed_measurements_type!='Observation'])
p_array[timeframed_measurements_type=='Observation']=monitoring_weight/len(
timeframed_measurements_type[timeframed_measurements_type=='Observation'])
for i in range(len(bootstrap_timeframed_measurements_xyz)):
    bootstrap_indices = np.random.choice(np.arange(len(timeframed_measurements_xyz)), 
                                         size=int(len(timeframed_measurements_xyz)*0.8),
                                        replace=False,
                                        p=p_array)
    bootstrap_timeframed_measurements_xyz[i] = timeframed_measurements_xyz[bootstrap_indices]
# %%
bootstrap_IDW = []
for i in tqdm(range(len(bootstrap_timeframed_measurements_xyz))):
    bootstrap_IDW.append(wl.tree(bootstrap_timeframed_measurements_xyz[i,:,:2],
                            bootstrap_timeframed_measurements_xyz[i,:,2]))

#%%
bootstrap_realizations = []
for i in tqdm(range(len(bootstrap_IDW))):
    z = bootstrap_IDW[i](xx_yy)
    bootstrap_realizations.append(z)

# %%
z_mean = np.array(bootstrap_realizations)
z_mean = np.reshape(z_mean, (n_realizations,len(x_),len(y_)))
z_mean = np.mean(z_mean, axis=0)

# %%
z_var = np.array(bootstrap_realizations)
z_var = np.sqrt(np.var(z_var, axis=0))
z_var = np.reshape(z_var, (len(x_),len(y_)))

# %%
fig, ax = plt.subplots(figsize=(15,15))
#CA_map.boundary.plot(ax=ax, zorder=3)
#plt.grid(zorder=2.5, color='k')
CV_map.boundary.plot(ax=ax, zorder=3)
plt.pcolormesh(xx, yy, z_mean, zorder=-2, cmap=cmc.batlow)
#plt.colorbar()
plt.colorbar().set_label(label='Groundwater Depth (ft)',size=20)
#plt.scatter(timeframed_measurements_lon, timeframed_measurements_lat, color='red', s=10)
plt.pcolormesh(xx, yy, CV_overlay, cmap=new_binary, zorder=-1)
plt.title('IDW interpolation: spring 2021 (mean of 10000 realizations)', fontsize=20)
plt.xlabel('Longitude', fontsize=20)
plt.ylabel('Latitude', fontsize=20)


# %%
fig, ax = plt.subplots(figsize=(15,15))
#CA_map.boundary.plot(ax=ax, zorder=3)
CV_map.boundary.plot(ax=ax, zorder=4)
plt.pcolormesh(xx, yy, z_var/z_mean, zorder=1, cmap=cmc.batlow)
plt.colorbar().set_label(label='Groundwater Depth Standard Deviation (ft)',size=20)
plt.pcolormesh(xx, yy, CV_overlay, cmap=new_binary, zorder=2)
#plt.scatter(timeframed_measurements_lon, timeframed_measurements_lat, color='red', s=10)
plt.scatter(timeframed_measurements_lon, timeframed_measurements_lat, color='red', s=10)
plt.title('IDW interpolation: spring 2021 variance', fontsize=20)
plt.xlabel('Longitude', fontsize=20)
plt.ylabel('Latitude', fontsize=20)
plt.grid(zorder=0)
#plt.savefig('OK_CV_only_vari.png', dpi=300)
    # %%


#%%
# Find ids for given coordinate
# Convert x y to meshgrid
coord = [-121.00000000000011, 37.50000000000004]

gwd_sel = wl.resistivity_value_from_xy(xx = xx, yy = yy,values = z_mean, coord = coord)
print(gwd_sel)