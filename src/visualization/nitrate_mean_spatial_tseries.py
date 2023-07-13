#%%
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, 'src')
import config

#%%
res_var = [
    'Resistivity_lyrs_9_rad_2_miles'
]

#%%

# Constants
rad_well = 2
gama_old_new = 2 # 1: earlier version, 2: latest version of GAMA
all_dom_flag = 1 # 1: All, 2: Domestic
if all_dom_flag == 2:
    well_type_select = {1: 'Domestic', 2: 'DOMESTIC'}.get(gama_old_new) 
else:
    well_type_select = 'All'

# Read dataset
def load_data(version):
    """Load data based on version"""
    filename = "Dataset_processed_GAMAlatest.csv" if version == 2 else "Dataset_processed.csv"
    df = pd.read_csv(config.data_processed / filename)
    return df

def filter_data(df, well_type,all_dom_flag):
    """Filter"""
    exclude_subregions = [14, 15, 10, 19, 18, 9, 6]
    if all_dom_flag == 2:
        df = df[df.well_type ==  well_type] 
    df = df[(df[f'thickness_abovCond_{round(.1*100)}_lyrs_9_rad_2miles'] <= 31) | (~df['SubRegion'].isin(exclude_subregions))]
    return df

# Load and process data
df_main = load_data(gama_old_new)
df = df_main[df_main.well_data_source == 'GAMA'].copy()

df['well_type_encoded'] = pd.factorize(df['well_type'])[0]
df['well_type_encoded'] = df['well_type_encoded'].where(df['well_type'].notna(), df['well_type'])

# separate wells inside cv
well_cv = pd.read_csv(config.data_processed / 'wells_inside_CV_GAMAlatest.csv',index_col=False)
# # Assuming df is your dataframe with all wells
df_cv = df[df['well_id'].isin(well_cv['well_id'])]

df = df_cv.copy() #filter_data(df_cv, well_type_select,all_dom_flag)
#%%
# Start with the columns you explicitly defined
columns_to_keep = ['well_id','SubRegion','APPROXIMATE LATITUDE', 'APPROXIMATE LONGITUDE']

# # Now add columns for every 3-year period from 1990 to 2022
for year in range(1990, 2022, 3):
    key = f"{year}-{year + 2}" # Adjusted the end year to make it a 5-year period without overlap
    columns_to_keep.append(f'mean_concentration_{key}')

# Now add columns for every 5-year period from 1990 to 2022
# for year in range(1990, 2022, 5):
#     key = f"{year}-{year + 4}" # Adjusted the end year to make it a 5-year period without overlap
#     columns_to_keep.append(f'mean_concentration_{key}')

# Filter the DataFrame
df2 = df[columns_to_keep]

# df2 = df2.dropna()
#%%
df2.to_csv(config.data_processed / 'nitrate_selectPeriodsmean_gamalatest.csv')
# %%
#==========================================
# Codes to create spatial plot
#==========================================
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rasterio.transform import from_origin
from matplotlib.animation import FuncAnimation
from PIL import Image
from shapely.geometry import box
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import io
import rasterio
from matplotlib.animation import PillowWriter

# %%
# Creating the 1km by 1km grid

# Load shapefile
cv = gpd.read_file(config.shapefile_dir / 'cv.shp')
ca = gpd.read_file(config.shapefile_dir / 'CA_State_TIGER2016.shp')
ca = ca.to_crs(cv.crs)
# Define grid size
grid_size = 0.05 # 0.01 approx equal to 1km

# Define the boundaries for your grid
x_min,y_min,x_max,y_max = cv.total_bounds

# Create the grid
x_grid = np.arange(x_min,x_max,grid_size)
y_grid = np.arange(y_min,y_max,grid_size)
grid = []

for x in x_grid:
    for y in y_grid:
        grid.append(gpd.GeoDataFrame({'geometry': gpd.GeoSeries(box(x, y, x+grid_size, y+grid_size))}))

# Concatenate all dataframes to a single one
grid = pd.concat(grid, ignore_index=True)


# %%
# Spatially averaging wells inside each grid
df2 = df2.rename(columns={'APPROXIMATE LATITUDE': 'latitude', 'APPROXIMATE LONGITUDE': 'longitude'})
# Convert df to a GeoDataFrame
gdf = gpd.GeoDataFrame(df2, geometry=gpd.points_from_xy(df2.longitude, df2.latitude))

# Ensure the GeoDataFrames have the same CRS (Coordinate Reference System)
gdf.crs = grid.crs

# Perform a spatial join
joined = gpd.sjoin(gdf, grid, how="inner", op='within')

# For each time period, calculate the mean concentration for each grid cell
mean_concentration_columns = [column for column in df2.columns if "mean_concentration" in column]

for column in mean_concentration_columns:
    # Create a GeoDataFrame with only the wells that have data for this time period
    gdf_time_period = gdf[gdf[column].notna()].copy()

    # Perform a spatial join
    joined = gpd.sjoin(gdf_time_period, grid, how="inner", op='within')

    # Calculate the mean concentration for each grid cell
    mean_concentrations = joined.groupby('index_right')[column].mean()

    # Add this data as a new column to the grid GeoDataFrame
    grid[column] = mean_concentrations

# Reset the index of the grid DataFrame
grid.reset_index(drop=True, inplace=True)


#%%
# Spatial plot of one time stamp
import matplotlib.pyplot as plt
import geopandas as gpd

#%%
import numpy as np

# Define bins and labels for discretization
bins = [0, 2, 4, 6, 8, 10, np.inf]
labels = ['0-2', '>2-4', '>4-6', '>6-8', '>8-10', '>10']

# Create a copy of the grid DataFrame
grid_discrete = grid.copy()

time_periods = [col for col in grid.columns if 'mean_concentration' in col]

# Apply discretization to each concentration column
for col in time_periods:
    grid_discrete[col] = pd.cut(grid[col], bins=bins, labels=labels)

# list of columns to check for NaN
mean_concentration_columns = [col for col in grid_discrete.columns if 'mean_concentration' in col]

# drop rows where all mean_concentration columns are NaN
gdf_discrete = grid_discrete.dropna(subset=mean_concentration_columns, how='all')


# %%
# Create the plot
fig, ax = plt.subplots(1, 1)
grid.plot(column='mean_concentration_2017-2019', ax=ax, legend=True, cmap='viridis')

ax.set_title("Mean Concentration 2015-2019")
ax.axis('off')

# Show the plot
plt.show()

# %%
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


fig, ax = plt.subplots(figsize=(4,7))

# Create a custom colormap
# colors = ["#D3D3D3", "#87CEEB", "#8BCB4A", "yellow", "orange", "red"]
# cmap = mcolors.LinearSegmentedColormap.from_list("", colors)
cmap = plt.cm.get_cmap('RdYlGn_r')


# List of columns representing the different time periods
time_periods = [col for col in grid_discrete.columns if 'mean_concentration' in col]

def update(num):
    ax.clear()
    grid_discrete.plot(column=time_periods[num], ax=ax, legend=True, cmap=cmap,alpha = .8, zorder = 2)
    
    # Add your shapefile to the plot
    cv.boundary.plot(ax=ax, color='black',facecolor='white', linewidth=1,zorder = 1)
    ca.boundary.plot(ax=ax, color='gray',facecolor='lightgray', linewidth=.5,zorder = 0,alpha = .5)
    
    ax.set_title(f"Mean Concentration {time_periods[num].split('_')[-1]}")
    ax.axis('off')

    # Set the plot extent to match the cv GeoDataFrame's geometry
    ax.set_xlim(cv.total_bounds[0]-.5, cv.total_bounds[2]+.5)
    ax.set_ylim(cv.total_bounds[1]-.5, cv.total_bounds[3]+.5)

ani = FuncAnimation(fig, update, frames=range(len(time_periods)), repeat=True, interval=1500)

# To display the animation in the Jupyter Notebook
HTML(ani.to_jshtml())


# %%
ani.save(f'/Users/szalam/Main/00_Research_projects/mean_concentration_{grid_size*100}km.gif', writer='imagemagick')

# %%
