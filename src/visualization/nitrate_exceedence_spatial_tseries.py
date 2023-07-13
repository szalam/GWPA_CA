#%%
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np

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
df2.iloc[0:50].to_csv(config.data_processed / 'nitrate_selectPeriodsmean_gamalatest.csv')
#%%
# Define the function to categorize the mean concentration
def categorize_concentration(concentration):
    if pd.isnull(concentration):
        return 0
    elif concentration <= 5:
        return 1
    elif concentration <= 10:
        return 2
    else:
        return 3

# Apply the function to each mean concentration column
for column in df2.columns:
    if 'mean_concentration' in column:
        df2[column + '_cat'] = df2[column].apply(categorize_concentration)

df2.head()

#%%
#==========================================
# Codes to create spatial plot
#==========================================


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
# Rename columns for easier handling
df2 = df2.rename(columns={'APPROXIMATE LATITUDE': 'latitude', 'APPROXIMATE LONGITUDE': 'longitude'})

# Create a GeoDataFrame with categorized concentrations
gdf_category = gpd.GeoDataFrame(df2, geometry=gpd.points_from_xy(df2.longitude, df2.latitude))

# Ensure the GeoDataFrames have the same CRS (Coordinate Reference System)
gdf_category.crs = grid.crs
#%%

# Now each well has a category for each time period, perform a spatial join
joined = gpd.sjoin(gdf_category, grid, how="inner", op='within')

#%%
grid_discrete = grid.copy()

# Add a new column 'grid_index' with the same values as the grid index
grid_discrete['grid_index'] = grid.index

#%%
# Define the function to assign grid values
def assign_grid_value(group, cat_column):
    counts = group[cat_column].value_counts()
    for i in [3, 2, 1, 0]:
        if counts.get(i, 0) >= 3:
            return i
    return None

# Initialize a new DataFrame to store the results
df_grid_scores = grid_discrete[['grid_index']].copy()

# List of category columns
cat_columns = [col for col in joined.columns if '_cat' in col]

# Assign grid values for each timestamp
for cat_column in cat_columns:
    timestamp = cat_column.replace('_cat', '')
    df_grid_scores[timestamp] = joined.groupby('index_right').apply(assign_grid_value, cat_column)
    df_grid_scores[timestamp] = df_grid_scores[timestamp].fillna(0)

df_grid_scores.head()

#%%


grid.iloc[0:100].to_csv(config.data_processed / 'grid.csv')
gdf_category.iloc[0:100].to_csv(config.data_processed / 'gdf_category.csv')
grid_discrete.iloc[0:100].to_csv(config.data_processed / 'grid_discrete.csv')
joined.iloc[0:100].to_csv(config.data_processed / 'joined.csv')

#%%
# Merge grid_discrete and df_grid_scores on grid_index
df_merged = pd.merge(grid_discrete, df_grid_scores, how='left', on='grid_index')

# Display the first few rows of the merged dataset
df_merged.head()

#%%
df_merged_label = df_merged.copy()
# List of columns representing the different time periods
time_periods = [col for col in df_merged.columns if 'mean_concentration' in col]

# Define the bins and labels again
bins = [-np.inf, 0, 1, 2, 3, np.inf]
labels = ['nan', '1', '2', '3', 'nan']

# Convert 0,1,2,3 into labels for each time period again
for time_period in time_periods:
    df_merged_label[time_period] = pd.cut(df_merged_label[time_period], bins=bins, labels=labels, ordered=False)

df_merged_label.head()

#%%
df_merged_label.replace('nan', np.nan, inplace=True)

#%%
# Create the plot
fig, ax = plt.subplots(1, 1)
df_merged_label.plot(column='mean_concentration_2020-2022', ax=ax, legend=True, cmap='viridis')

ax.set_title("Mean Concentration 2020-2022")
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
time_periods = [col for col in df_merged_label.columns if 'mean_concentration' in col]

def update(num):
    ax.clear()
    df_merged_label.plot(column=time_periods[num], ax=ax, legend=True, cmap=cmap,alpha = .8, zorder = 2)
    
    # Add your shapefile to the plot
    cv.boundary.plot(ax=ax, color='black',facecolor='white', linewidth=1,zorder = 1)
    ca.boundary.plot(ax=ax, color='gray',facecolor='lightgray', linewidth=.5,zorder = 0,alpha = .5)
    
    ax.set_title(f"Concentration levels during {time_periods[num].split('_')[-1]}")
    ax.axis('off')

    # Set the plot extent to match the cv GeoDataFrame's geometry
    ax.set_xlim(cv.total_bounds[0]-.5, cv.total_bounds[2]+.5)
    ax.set_ylim(cv.total_bounds[1]-.5, cv.total_bounds[3]+.5)

ani = FuncAnimation(fig, update, frames=range(len(time_periods)), repeat=True, interval=1500)

# To display the animation in the Jupyter Notebook
HTML(ani.to_jshtml())


# %%
ani.save(f'/Users/szalam/Main/00_Research_projects/risk_scores_{grid_size*100}km.gif', writer='imagemagick')

# %%
