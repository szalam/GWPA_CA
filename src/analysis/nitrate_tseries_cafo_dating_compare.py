#%%
import sys
sys.path.insert(0,'src')
sys.path.insert(0,'src/data')
import config
import pandas as pd
import ppfun as dp
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt 
from scipy import stats
import numpy as np

# CAFO dating data 
cafo_dts = pd.read_csv(config.data_processed / "CAFO_dating/CA_final_with_dates.csv")
# cafo_dts_gdf = gpd.GeoDataFrame(cafo_dts, geometry=gpd.points_from_xy(cafo_dts.longitude, cafo_dts.latitude))

# Convert latitude and longitude columns to shapely Points
geometry = [Point(xy) for xy in zip(cafo_dts['longitude'], cafo_dts['latitude'])]

# Create a GeoDataFrame
crs = {'init': 'epsg:4326'} # CRS stands for Coordinate Reference System, this line specify the system
cafo_dts_gdf = gpd.GeoDataFrame(cafo_dts, crs=crs, geometry=geometry)
# cafo_dts_gdf.to_file(config.data_processed / 'kml' / f"CAFO_shape.shp", driver='ESRI Shapefile')


# Read region boundary
cv = dp.get_region(config.shapefile_dir   / "cv.shp")


#==================== User Input requred ==========================================
rad_buffer     = 2              # well buffer radius in miles
#=================================================================================

# # Convert the CRS of the GeoDataFrame to a projected CRS (e.g. UTM zone 10N)
gdf_buffer = cafo_dts_gdf.to_crs({'init': 'epsg:32610'})

# # Create a buffer of 2 miles in the projected CRS
gdf_buffer.geometry = gdf_buffer.buffer(1609.345 * rad_buffer)

# # Convert the CRS back to WGS 84
gdf_buffer = gdf_buffer.to_crs({'init': 'epsg:4326'})

# Perform  clip on the well data based on selected boundary
gdf_cafobuffer = gpd.sjoin(gdf_buffer, cv, how='inner', op='within')

gdf_cafobuffer = gdf_cafobuffer[['Facility_Name','Facility_Address','SIC_NAICS','Cafo_Population',
 'constructions_lower', 'constructions_higher', 'geometry','FID']]
#%%

# Read dataset
df = pd.read_csv(config.data_processed / "Dataset_processed.csv")
df = df[df.well_data_source == 'GAMA']
df = df[df.measurement_count > 30]
#
# if 'APPROXIMATE LONGITUDE' in df:
#     # Create a example dataframe with lat and lon columns
#     # df = pd.DataFrame({'well_id':df['WELL ID'],'lat':df['APPROXIMATE LATITUDE'], 'lon':df['APPROXIMATE LONGITUDE']})
#     df.rename(columns = {'APPROXIMATE LATITUDE': 'lat', 'APPROXIMATE LONGITUDE':'lon'}, inplace = True)
# if 'WELL ID' in df:
#     # Create a example dataframe with lat and lon columns
#     df.rename(columns = {'WELL ID': 'well_id'}, inplace = True)

df.rename(columns = {'APPROXIMATE LATITUDE': 'lat', 'APPROXIMATE LONGITUDE':'lon', 'WELL ID': 'well_id'}, inplace = True)

# Convert latitude and longitude columns to shapely Points
geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
crs = {'init': 'epsg:4326'}
gdf2 = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
gdf2 = gdf2[['well_id','geometry']]

#%%
cafobuffer_well_intersect = gpd.overlay(gdf2,gdf_cafobuffer,how='intersection')

# %%
well_src = 'GAMA'
#=========================== Import water quality data ==========================
# Read water quality data
if well_src == 'GAMA':
    file_polut = config.data_gama_all / 'CENTRALVALLEY_NO3N_GAMA.csv'
    df_polut= dp.get_polut_df(file_sel = file_polut)
    df_polut.rename(columns = {'GM_WELL_ID':'well_id', 'GM_LATITUDE':'lat', 'GM_LONGITUDE':'lon', 'GM_CHEMICAL_VVL': 'CHEMICAL', 'GM_RESULT': 'RESULT','GM_WELL_CATEGORY':'DATASET_CAT','GM_SAMP_COLLECTION_DATE':'DATE'}, inplace = True)
    df_polut['DATE']= pd.to_datetime(df_polut['DATE'])
elif well_src == 'UCD':
    file_polut = config.data_raw / "nitrate_data/UCDNitrateData.csv"
    df_polut = dp.get_polut_df(file_sel = file_polut)
    df_polut.rename(columns = {'WELL ID':'well_id'}, inplace = True)

#%%
# Convert dates to datetime format
cafobuffer_well_intersect['constructions_lower'] = pd.to_datetime(cafobuffer_well_intersect['constructions_lower'], format='%Y-%m-%d')
cafobuffer_well_intersect['constructions_higher'] = pd.to_datetime(cafobuffer_well_intersect['constructions_higher'], format='%Y-%m-%d')
df_polut['DATE'] = pd.to_datetime(df_polut['DATE'], format='%Y-%m-%d')

# merge the two dataframes on well_id
merged = pd.merge(cafobuffer_well_intersect, df_polut, on='well_id')

#%%
construction_select = 'constructions_higher'

def significant_change(x):
    t, p = stats.ttest_ind(x[x['DATE'] >= x[f'{construction_select}']]['RESULT'].dropna(), x[x['DATE'] < x[f'{construction_select}']]['RESULT'].dropna(), equal_var=False)
    if p < 0.05:
        return 'significant'
    else:
        return 'not-significant'

#%%
# calculate mean of RESULT for each well_id, grouping by constructions_higher before and after
result = merged.groupby(['well_id', 'constructions_higher','Facility_Address']).apply(lambda x: x[x['DATE'] >= x['constructions_higher']]['RESULT'].mean() - x[x['DATE'] < x['constructions_higher']]['RESULT'].mean())

# create a dataframe from the result series
df_result = result.reset_index().rename(columns={0: 'change_in_nitrate_concentration'})
df_result = df_result.drop_duplicates(subset='well_id', keep='first')

# add constructions_lower to df_result
df_result = df_result.merge(merged[['well_id', 'constructions_higher', 'constructions_lower','Facility_Address']].drop_duplicates(subset=['well_id', 'Facility_Address'], keep='first'), on=['well_id', 'Facility_Address'])

def significant_change(x):
    t, p = stats.ttest_ind(x[x['DATE'] >= x['constructions_higher']]['RESULT'].dropna(), x[x['DATE'] < x['constructions_higher']]['RESULT'].dropna(), equal_var=False)
    if p < 0.05:
        return 'significant'
    else:
        return 'not-significant'

result = merged.groupby(['well_id', 'constructions_higher','Facility_Address']).apply(significant_change)
df_result['significance'] = result.reset_index(drop=True)

if 'constructions_higher_y' in df_result.columns:
    df_result.drop('constructions_higher_y', axis=1, inplace=True)

# df_result.dropna(inplace=True)
df_result_increase = df_result[df_result['change_in_nitrate_concentration'] > 0].reset_index(drop=True)
df_result_increase = df_result[df_result['change_in_nitrate_concentration'] > 0].rename(columns={'constructions_higher_x': 'constructions_higher'})
df_result_decrease = df_result[df_result['change_in_nitrate_concentration'] < 0].reset_index(drop=True)
df_result_decrease = df_result[df_result['change_in_nitrate_concentration'] < 0].rename(columns={'constructions_higher_x': 'constructions_higher'})

#%%
#===================================================================================================================
# Plot time series of nitrate of wells having positive correlation between nitrate and CAFO but close conductivity
#===================================================================================================================

def plot_time_series(df, ucd_ids, df_result_delta, plt_row = 7, plt_colm = 4):
    num_plots = len(ucd_ids)
    # num_canvases = (num_plots + 8) // 9 # 3x3
    num_canvases = (num_plots + (plt_row*plt_colm-1)) // (plt_row*plt_colm) # 5x5
    
    for canvas_num in range(num_canvases):
        fig, ax = plt.subplots(plt_row, plt_colm, figsize=(29, 16))
        ax = ax.flatten()
        for plot_num in range(plt_row*plt_colm):
            index = canvas_num * (plt_row*plt_colm) + plot_num  
            if index >= num_plots:
                break
            ucd_id = ucd_ids[index]
            well_id = well_id = f'{ucd_id}' # if using UCD: f'NO3_{ucd_id}'
            dp.get_plot_time_series_well_canvas(df, well_id = well_id, xvar = 'DATE', yvar = 'RESULT', ax=ax[plot_num])
            constructions_higher = df_result_delta.loc[df_result_delta['well_id'] == well_id, f'{construction_select}'].iloc[0]
            ax[plot_num].axvline(x=constructions_higher, color='red')
            
            constructions_lower = df_result_delta.loc[df_result_delta['well_id'] == well_id, 'constructions_lower'].iloc[0]
            if pd.isna(constructions_lower) is not True:
                ax[plot_num].axvline(x=constructions_lower, color='blue')
                ax[plot_num].axvspan(constructions_lower, constructions_higher, color='gray', alpha=0.2)
            # print(constructions_lower)
            ax[plot_num].set_xlim(pd.Timestamp('1992-01-01'), pd.Timestamp('2022-12-31'))
        plt.tight_layout()
        plt.show()


# positive_change_mask = df_result['change_in_nitrate_concentration'] > 0
df_result_increase_tmp = df_result_increase[df_result_increase.significance == 'significant'] #.tail(1000)
# removing cases where no construction_lower
df_result_increase_tmp = df_result_increase_tmp[pd.isna(df_result_increase_tmp.constructions_lower) == False]
# Plot time series for well_id with positive change_in_nitrate_concentration
plot_time_series(df=df_polut, ucd_ids=df_result_increase_tmp['well_id'].tolist(), df_result_delta=df_result_increase_tmp, plt_row=6, plt_colm=5)
#%%