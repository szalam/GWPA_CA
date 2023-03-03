#================================================================================
# This script analyzies the relationship between animal population and nitrate
# for wells when they have almost similar conductivity
#================================================================================
#%%
import sys
sys.path.insert(0,'src')
sys.path.insert(0, 'src/data')

import pandas as pd
import config
import numpy as np
import matplotlib.pyplot as plt 
import ppfun as dp
from scipy import stats
import seaborn as sns
import matplotlib as mpl
import get_infodata as gi
import geopandas as gpd

# Read dataset
df_main = pd.read_csv(config.data_processed / "Dataset_processed.csv")
#%%
df = df_main.copy()
plt_cond = 0.05      # Threshold condoctivity above which thickness calculated
lyrs = 4
#%%
df = df[df.well_data_source == 'GAMA']
# df = df[df['SubRegion'] != 14]
# df = df[df['SubRegion'] == 10]
# df = df[df.measurement_count > 4]
# df = df[df.city_inside_outside == 'outside_city']
well_type_select = 'Domestic' # 'Water Supply, Other', 'Municipal', 'Domestic'
df = df[df.well_type ==  well_type_select] 
# df = df[pd.isna(df.GWPAType)==True] # False for areas with GWPA

# Remove high salinity regions
exclude_subregions = [14, 15, 10, 19,18, 9,6]
# filter aemres to keep only the rows where Resistivity is >= 10 and SubRegion is not in the exclude_subregions list
df = df[(df[f'thickness_abovCond_{round(plt_cond*100)}_lyrs_9'] <= 31) | (~df['SubRegion'].isin(exclude_subregions))]
# df = df[(df[f'thickness_abovCond_{round(0.1*100)}_lyrs_{lyrs}'] == 0)]
# df = df[(df[f'thickness_abovCond_{round(0.08*100)}_lyrs_{lyrs}'] == 0)]
        
# condition = (df[f'thickness_abovCond_{round(plt_cond*100)}'] > 31) & (df['mean_nitrate'] > 0)
# df = df[condition==False]

# df = df[df.mean_nitrate>10]
# df = df.dropna()

layer_depths = gi.dwr_aem_depths()

#%%
cond_threshold_fine_list = [0.01,0.02,0.03, 0.05,0.06,0.07, 0.08, 0.1, 0.13, 0.15, 0.18, 0.2, 0.25]
#%%

plt.scatter(df[f'thickness_abovCond_{round(plt_cond*100)}_lyrs_{lyrs}'], df.mean_nitrate, s = 1.5)
plt.ylim(0 ,40)
# plt.xlim(0 ,100)
# plt.title(f'Well type: {well_type_select}')
plt.xlabel(f'Thickness of layers with resistivity <= {1/plt_cond}')
plt.ylabel('Nitrate-N')
# plt.colorbar(label='Topsoil conductivity')

#%%
import statsmodels.api as sm
x = df[f'thickness_abovCond_{round(plt_cond*100)}_lyrs_{lyrs}']
y = df.mean_nitrate

lowess = sm.nonparametric.lowess(y, x, frac=.2) # adjust frac to control smoothing
plt.scatter(x, y, s=1.5)
plt.plot(lowess[:,0], lowess[:,1], c='r')
plt.ylim(0, 40)
plt.xlabel(f'Thickness of layers with resistivity <= {1/plt_cond}')
plt.ylabel('Nitrate-N')
plt.show()

#%%
# Boxplot
# create a new column in the dataframe to map depth values to range categories
df['Depth_range'] = pd.cut(df[f'thickness_abovCond_{round(plt_cond*100)}_lyrs_{lyrs}'], 
                        #    bins=[-1, 0, 5, 15, 25, 35], 
                        #    bins=[-1, 0, 5, 10, 15, 25, 35], 
                        #    bins=[-1, 0, 10, 25, 35], 
                        #    labels=['0', '(0,5]', '(5,15]', '(15,25]', '(25,35)'])
                            # labels=['0', '(0,5]', '(5,10]', '(10,15]', '(15,25]', '(25,35]'])
                            # labels=['0', '(0,10]', '(10,25]','(25,35]'])
                              bins=[-1, 0, 3, 6, 10],
                              labels=['0', '(0,3]', '(3,6]','(6,10]'])


# create the boxplot using seaborn
sns.boxplot(x='Depth_range', y='mean_nitrate', data=df, 
            palette='viridis', linewidth=1)

# set the y-axis limits
plt.ylim(0, 40)

# set the title and axis labels
plt.title(f'Well type: {well_type_select}')
plt.xlabel(f'Thickness of layers with resistivity <= {1/plt_cond}')
plt.ylabel('Nitrate-N')

# display the plot
plt.show()







#%% 
# Export high depth and high nitrate well locations
# Create a geodataframe with lat, lon, and well_id columns
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['APPROXIMATE LONGITUDE'], df['APPROXIMATE LATITUDE']))

# Define the condition for selecting rows
condition = (df[f'thickness_abovCond_{round(plt_cond*100)}'] > 30) & (df['mean_nitrate'] > 10)

# Apply the condition to the geodataframe
subset_gdf = gdf[condition]

# Export the subset as a shapefile
subset_gdf.to_file(config.data_processed / 'highthickness_high_nitrate/highthickness_high_nitrate.shp', driver='ESRI Shapefile')




#%%
# Density plot
# #%%
# fig, ax = plt.subplots(figsize=(8, 6))

# for thickness in range(0, 41, 10):
#     df_range = df[(df[f'Depth_above_condthres_{round(plt_cond*100)}'] >= thickness) & (df[f'Depth_above_condthres_{round(plt_cond*100)}'] < thickness+10)]
#     nitrate_values = df_range.mean_nitrate.values
#     ax.hist(nitrate_values, alpha=0.5, label=f'{thickness}m <= Thickness < {thickness+10}m')

# ax.set_xlabel('Nitrate-N')
# ax.set_ylabel('Frequency')
# plt.ylim(0, 60)
# ax.set_title(f'Nitrate distribution for different thickness intervals\nWell type: {well_type_select}, Conductivity threshold: {plt_cond}')
# ax.legend()
# plt.show()

# #%%
# import seaborn as sns

# depth_breaks = [0,1,5,10,15,20,25,30,40]
# # Plot histogram with density curve
# fig, ax = plt.subplots(figsize=(8, 6))
# colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
# for i,thickness in enumerate(depth_breaks):
#     df_range = df[(df[f'Depth_above_condthres_{round(plt_cond*100)}'] >= thickness) & (df[f'Depth_above_condthres_{round(plt_cond*100)}'] < depth_breaks[i+1])]
#     nitrate_values = df_range.mean_nitrate.values
#     sns.histplot(nitrate_values, label=f'{thickness}m <= Thickness < {depth_breaks[i+1]}m',kde=True, alpha=1, bins=np.arange(0, 100, 1), color = colors[i],fill=False,linewidth = 0)
#     if thickness == 30:
#         break

# plt.ylim(0,20)
# plt.xlim(0,100)
# ax.set_xlabel('Nitrate-N')
# ax.set_ylabel('Density')
# ax.set_title(f'Nitrate distribution for thickness above and below 5 meters\nWell type: {well_type_select}, Conductivity threshold: {plt_cond}')
# ax.legend()
# plt.show()



# # %%
# plt.scatter(df[f'Cond_x_Depth_above_condthres_{round(plt_cond*100)}'], df.mean_nitrate, s = 1.5, c=df['Conductivity_depthavg_upto_layer_1'], cmap='viridis', vmin=0, vmax=0.15)
# plt.ylim(0 ,30)
# # plt.xlim(0 ,100)
# # plt.xlabel(f'{aem_type}')
# plt.ylabel('Nitrate-N')
# plt.xlabel(f'Thickness of layers with conductivity > {plt_cond}')
# plt.colorbar(label='Topsoil conductivity')
# # %%
# plt.scatter(df['Conductivity_lyrs_9'], df['Conductivity_lyrs_1'])











#==================================================================================
# Code to calculate clay thickness (high conductivity thickness) upto gw level
# Note: there are smaller number of locations having groundwater depth data
#==================================================================================
#==================================================================================
# Now determining conductivity upto groundwater depth, not fixed 30 meter
num_layers_to_gw_list = []
for index, row in df.iterrows():
    if pd.isnull(row['gwdep']):
        num_layers_to_gw_list.append(np.nan)
        continue
        
    gwdep_m = row['gwdep'] / 3.28  # depth to groundwater in meters
    
    num_layers = len(layer_depths)  # set to maximum number of layers by default
    
    for i, layer_depth in enumerate(layer_depths):
        if sum(layer_depths[:i+1]) >= gwdep_m:
            num_layers = i+1
            break
    
    num_layers_to_gw_list.append(num_layers)
    
df['num_layers_to_gw'] = num_layers_to_gw_list

df['num_layers_to_gw'].describe()
#%%
df2 = df.copy()
df2 = df2[pd.isnull(df2['num_layers_to_gw']) == False]

# divide the Conductivity_depthwtd_lyr value columns with the corresponding layer depths for each row and layer
for i in range(1, int(max(df2['num_layers_to_gw']))+1):
    df2[f'Conductivity_lyr{i}'] = df2.apply(lambda row, i=i: row[f'Conductivity_depthwtd_lyr{i}'] / layer_depths[i-1] if i <= row['num_layers_to_gw'] else 0, axis=1)

#%%
# calculate the total depth of layers having conductivity greater than cond_threshold_fine for each row and store the depths in a new column
for i in range(1, 20):
    df2[f'Conductivity_lyr{i}'] = df2[f'Conductivity_depthwtd_lyr{i}'] / layer_depths[i-1]

for cond_threshold_fine in cond_threshold_fine_list:
    # calculate the total depth of layers having conductivity greater than cond_threshold_fine for each well_id and store the depths in a new column
    column_name = f'Depth_above_condthres_{round(cond_threshold_fine*100)}'
    df2[column_name] = 0
    
    for index, row in df2.iterrows():
        depth_above_threshold = 0
        for i in range(1, int(row['num_layers_to_gw'])+1):
            if row[f'Conductivity_lyr{i}'] > cond_threshold_fine:
                depth_above_threshold += layer_depths[i-1]
        df2.at[index, column_name] = depth_above_threshold

    # calculate the sum of conductivities x layer depths for layers separated in the above step
    # df2[f'Cond_x_Depth_above_condthres_{round(cond_threshold_fine*100)}'] = df2.apply(lambda row: sum([row[f'Conductivity_lyr{i}'] * layer_depths[i-1] for i in range(1, row['num_layers_to_gw']+1) if row[f'Conductivity_lyr{i}'] > cond_threshold_fine]), axis=1)
#================================================================================
#%%
plt_cond = 0.05
# plt.scatter(df['Depth_above_condthres_8'], df.mean_nitrate, s = 1.5, c = 'red')
plt.scatter(df2[f'Depth_above_condthres_{round(plt_cond*100)}'], df2.mean_nitrate, s = 1.5, c='red')
plt.ylim(0 ,100)
# plt.xlim(0 ,100)
plt.title(f'Well type: {well_type_select}')
plt.xlabel(f'Thickness of layers with conductivity > {plt_cond}')
plt.ylabel('Nitrate-N')
# plt.colorbar(label='Topsoil conductivity')

#===================== Conductivity 

#%%
layers_considered = 9 # fixed to around 30 m depth (9 layers)
for cond_threshold_fine in cond_threshold_fine_list:
    # divide the Conductivity_depthwtd_lyr value columns with the corresponding layer depths
    for i in range(1, layers_considered+1):
        df[f'Conductivity_lyr{i}'] = df[f'Conductivity_depthwtd_lyr{i}'] / layer_depths[i-1]

    # calculate the total depth of layers having conductivity greater than cond_threshold_fine for each well_id and store the depths in a new column
    df[f'Depth_above_condthres_{round(cond_threshold_fine*100)}'] = df.apply(lambda row: sum([layer_depths[i-1] for i in range(1, layers_considered+1) if row[f'Conductivity_lyr{i}'] > cond_threshold_fine]), axis=1)

    # calculate the sum of conductivities x layer depths for layers separated in the above step
    df[f'Cond_x_Depth_above_condthres_{round(cond_threshold_fine*100)}'] = df.apply(lambda row: sum([row[f'Conductivity_lyr{i}'] * layer_depths[i-1] for i in range(1, layers_considered+1) if row[f'Conductivity_lyr{i}'] > cond_threshold_fine]), axis=1)

#%%
# Calculate depth average conductivity
num_layers = [1,9]  # consider the first 9 layers
for lyrs in num_layers:
    numerator_cols = [f'Conductivity_depthwtd_lyr{i}' for i in range(1, lyrs+1)]
    denominator_vals = sum(layer_depths[:lyrs])

    df[f'Conductivity_depthavg_upto_layer_{lyrs}'] = df[numerator_cols].sum(axis=1) / denominator_vals
