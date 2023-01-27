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
import geopandas as gpd
import matplotlib.pyplot as plt 
from tqdm import tqdm
import ppfun as dp

from scipy.stats import ttest_ind
# from visualize import vismod

# Read dataset
df = pd.read_csv(config.data_processed / "Dataset_processed.csv")
df = df[df.well_data_source == 'UCD']


df = df[['well_id','Conductivity','measurement_count','mean_nitrate','area_wt_sagbi', 'Cafo_Population_5miles','Average_ag_area','change_per_year','total_ag_2010','APPROXIMATE LATITUDE', 'APPROXIMATE LONGITUDE','city_inside_outside']]
df = df[df.measurement_count > 4]
# df = df[df.city_inside_outside == 'outside_city']
# df = df.head(1000)

#get the unique well ids
well_ids = df["well_id"].unique()

#======================================================================
# Plotting function
#======================================================================
def scatter_twovars(df, selected_well_ids, xvar = "Cafo_Population_5miles", yvar ="mean_nitrate",
                    xlab = "Animal population", ylab = "Nitrate concentration"):
    # separate data from df using those well_id
    selected_data = df[df["well_id"].isin(selected_well_ids)]

    # scatterplot the nitrate and animal polulation for those well_ids
    plt.scatter(selected_data[xvar], selected_data[yvar])
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.show()

def stat_of_notdiff_wells(df, not_diff, high_corr, var_stat):
    cond_all = []
    for i in range(len(high_corr)):
        tmp = not_diff[high_corr[i]]
        selected_data = df[df["well_id"].isin(tmp)]
        cond_mean = selected_data[f'{var_stat}'].mean()
        cond_all.append(cond_mean)
    return cond_all


def plot_point_shape(df, shapefile, not_diff, high_corr):
    # read in the shapefile and create a GeoDataFrame
    cv = gpd.read_file(shapefile)
    tmp = not_diff[high_corr]
    selected_data = df[df["well_id"].isin(tmp)]
    # create a GeoDataFrame from the df with latitude and longitude columns
    wells = gpd.GeoDataFrame(selected_data, geometry=gpd.points_from_xy(selected_data['APPROXIMATE LONGITUDE'], selected_data['APPROXIMATE LATITUDE']))
    # plot the wells
    ax1 = wells.plot(marker='o', color='red', markersize=5)
    # overlay the shapefile on top of the wells
    cv.plot(alpha=0.5, ax = ax1)
    plt.show()

#======================================================================
# Scenario 1: Wells located in similar conductivity zones. Select wells 
# having very close conductivity values
#======================================================================
# User input
min_sample_for_corr     = 10      # Min no of sample used for calculating correlation
conductiviy_threshold   = 0.00009 # Conductivity value +- range considered to estimate similar conductivity zones
corr_thresh_for_accept  = .75     # Correlation threshold above which samples are considered to have positive correlation

#create an empty list to store the well ids with similar conductivity
not_diff = []

#iterate over the well ids
for well_id in tqdm(well_ids):
    # get the conductivity of the current well
    cond = df.query("well_id == @well_id")["Conductivity"].mean()

    # select all the rows that have conductivity within +-0.01 of the current well ID's conductivity
    data = df.query(f"well_id != @well_id and @cond - {conductiviy_threshold} <= Conductivity <= @cond + {conductiviy_threshold}")
    well_id_list = list(data["well_id"].unique())

    if not_diff:
            not_diff.append(well_id_list)
    else:
        not_diff = [well_id_list]

#======================================================================
# Test 1: Compare cafo population vs nitrate concentration.
#======================================================================
# %%
# Loop through all wells with similar conductivity values to see if there 
# is a posirivity correlation between nitrate and cafo values

def corr_process(df,xvar = "mean_nitrate", yvar = "Cafo_Population_5miles", 
                min_sample_for_corr = min_sample_for_corr, corr_thresh_for_accept = corr_thresh_for_accept):
    
    high_corr_i = []
    cor_pos = 0
    for i in tqdm(range(len(not_diff))):

        if len(not_diff[i])>min_sample_for_corr:
            # select a list item in not_diff that has well_id
            selected_well_ids = not_diff[i]

            # separate data from df using those well_id
            selected_data = df[df["well_id"].isin(selected_well_ids)]

            # calculate the correlation between nitrate and animal population
            correlation = selected_data[xvar].corr(selected_data[yvar])

            # check if the correlation is positive
            if abs(correlation) > corr_thresh_for_accept:
                cor_pos += 1
                high_corr_i.append(i)   

    return cor_pos, high_corr_i

#%%
# Getting nuber of cases with correlation > corr_thresh_for_accept
cor_pos_test1, high_corr_test1 = corr_process(df,xvar = "mean_nitrate", yvar = "Cafo_Population_5miles", 
                min_sample_for_corr = min_sample_for_corr, corr_thresh_for_accept = corr_thresh_for_accept)
print(cor_pos_test1)

# Total highly correlated station lists
len(high_corr_test1)
#%%
# Plot scatter of CAFO population vs nitrate concentration
scatter_twovars(df, selected_well_ids = not_diff[high_corr_test1[33]], xvar = "Cafo_Population_5miles", yvar ="mean_nitrate",
                    xlab = "Animal population", ylab = "Nitrate concentration")
#%%
cond_vals = stat_of_notdiff_wells(df, not_diff = not_diff, high_corr = high_corr_test1, var_stat='Conductivity')
plt.hist(cond_vals)
#%%
df_trend_exists = df.loc[df['change_per_year'] != 0]
df_trend_exists.change_per_year.describe()
cond_vals = stat_of_notdiff_wells(df_trend_exists, not_diff = not_diff, high_corr = high_corr_test1, var_stat='change_per_year')
plt.hist(cond_vals)

#%%
df_gama = dp.get_raw_nitrate_data(data_source = 'GAMA')
#%%
dp.get_plot_time_series_well(df = df_gama, well_id = str(df_trend_exists.well_id.iloc[5]), xvar = 'DATE', yvar = 'RESULT')

#%%
cond_vals = stat_of_notdiff_wells(df, not_diff = not_diff, high_corr = high_corr_test1, var_stat='area_wt_sagbi')
plt.hist(cond_vals)
#%%
cond_vals = stat_of_notdiff_wells(df, not_diff = not_diff, high_corr = high_corr_test1, var_stat='Average_ag_area')
plt.hist(cond_vals)
#%%
plot_point_shape(df, shapefile = (config.shapefile_dir   / "cv.shp"), not_diff = not_diff, high_corr = high_corr_test1[49])
#%%
#==================================================================
# Test 2: Compare agricultural land use vs nitrate concentration
#==================================================================
# Getting nuber of cases with correlation > corr_thresh_for_accept
cor_pos_test2, high_corr_test2 = corr_process(df,xvar = "mean_nitrate", yvar = "Average_ag_area", 
                min_sample_for_corr = min_sample_for_corr, corr_thresh_for_accept = corr_thresh_for_accept)
print(cor_pos_test2)

# Total highly correlated station lists
len(high_corr_test2)
#%%
# Plot scatter of CAFO population vs nitrate concentration
scatter_twovars(df, selected_well_ids = not_diff[high_corr_test2[1]], xvar = "Average_ag_area", yvar ="mean_nitrate",
                    xlab = "Average Ag area", ylab = "Nitrate concentration")

#%%
cond_vals = stat_of_notdiff_wells(df, not_diff = not_diff, high_corr = high_corr_test2, var_stat='Conductivity')
plt.hist(cond_vals)
#%%
cond_vals = stat_of_notdiff_wells(df, not_diff = not_diff, high_corr = high_corr_test2, var_stat='area_wt_sagbi')
plt.hist(cond_vals)

cond_vals = stat_of_notdiff_wells(df, not_diff = not_diff, high_corr = high_corr_test2, var_stat='Average_ag_area')
plt.hist(cond_vals)
#%%
#==================================================================
# Test 3: Compare sagbi rating vs nitrate concentration
#==================================================================
# Getting nuber of cases with correlation > corr_thresh_for_accept
cor_pos_test3, high_corr_test3 = corr_process(df,xvar = "area_wt_sagbi", yvar = "mean_nitrate", 
                min_sample_for_corr = min_sample_for_corr, corr_thresh_for_accept = corr_thresh_for_accept)
print(cor_pos_test3)

# Total highly correlated station lists
len(high_corr_test3)
#%%
# Plot scatter of CAFO population vs nitrate concentration
scatter_twovars(df, selected_well_ids = not_diff[high_corr_test3[5]], xvar = "area_wt_sagbi", yvar ="mean_nitrate",
                    xlab = "SAGBI rating", ylab = "Nitrate concentration")

#%%
cond_vals = stat_of_notdiff_wells(df, not_diff = not_diff, high_corr = high_corr_test3, var_stat='Conductivity')
plt.hist(cond_vals)

#%%
plot_point_shape(df, shapefile = (config.shapefile_dir   / "cv.shp"), not_diff = not_diff, high_corr = high_corr_test3[5])


#%%
#======================================================================
# Scenario 2: Wells having similar Agricultural area in the buffer. 
#======================================================================
# User input
min_sample_for_corr     = 10      # Min no of sample used for calculating correlation
area_threshold          = 400     # Area value +- range considered to estimate similar area zones
corr_thresh_for_accept  = .75     # Correlation threshold above which samples are considered to have positive correlation


# Create an empty list to store the well ids with similar conductivity
not_diff = []

#iterate over the well ids
for well_id in tqdm(well_ids):
    # get the conductivity of the current well
    cond = df.query("well_id == @well_id")["Average_ag_area"].mean()

    # select all the rows that have conductivity within +-0.01 of the current well ID's conductivity
    data = df.query(f"well_id != @well_id and @cond - {area_threshold} <= Average_ag_area <= @cond + {area_threshold}")
    well_id_list = list(data["well_id"].unique())

    if not_diff:
            not_diff.append(well_id_list)
    else:
        not_diff = [well_id_list]

#%%
#==================================================================
# Test 4: Compare the conductivity vs nitrate concentration.
#==================================================================

# Getting nuber of cases with correlation > corr_thresh_for_accept
cor_pos_test4, high_corr_test4 = corr_process(df,xvar = "mean_nitrate", yvar = "Conductivity", 
                min_sample_for_corr = min_sample_for_corr, corr_thresh_for_accept = corr_thresh_for_accept)
print(cor_pos_test4)

# Total highly correlated station lists
len(high_corr_test4)

#%%
# Plot scatter of conductivity vs nitrate concentration
scatter_twovars(df, selected_well_ids = not_diff[high_corr_test4[0]], xvar = "Conductivity", yvar ="mean_nitrate",
                    xlab = "Conductivity", ylab = "Nitrate concentration")

#%%
#==================================================================
# Test 5: Compare the CAFO population vs nitrate concentration.
#==================================================================
# Getting nuber of cases with correlation > corr_thresh_for_accept
cor_pos_test5, high_corr_test5 = corr_process(df,xvar = "Cafo_Population_5miles", yvar = "mean_nitrate", 
                min_sample_for_corr = min_sample_for_corr, corr_thresh_for_accept = corr_thresh_for_accept)
print(cor_pos_test5)

# Total highly correlated station lists
len(high_corr_test5)
# %%
#%%
# Plot scatter of CAFO population vs nitrate vs nitrate concentration
scatter_twovars(df, selected_well_ids = not_diff[high_corr_test5[0]], xvar = "Cafo_Population_5miles", yvar ="mean_nitrate",
                    xlab = "Conductivity", ylab = "Nitrate concentration")

#%%
#===========================================================================
# Scenario 3: Wells located in similar SAGBI rating (surface soil property)
#===========================================================================
# User input
min_sample_for_corr     = 10      # Min no of sample used for calculating correlation
sagbi_threshold         = 7       # Sagbi value +- range considered to estimate similar sagbi zones
corr_thresh_for_accept  = .75     # Correlation threshold above which samples are considered to have positive correlation


# Create an empty list to store the well ids with similar conductivity
not_diff = []

#iterate over the well ids
for well_id in tqdm(well_ids):
    # get the conductivity of the current well
    cond = df.query("well_id == @well_id")["area_wt_sagbi"].mean()

    # select all the rows that have conductivity within +-0.01 of the current well ID's conductivity
    data = df.query(f"well_id != @well_id and @cond - {sagbi_threshold} <= area_wt_sagbi <= @cond + {sagbi_threshold}")
    well_id_list = list(data["well_id"].unique())

    if not_diff:
            not_diff.append(well_id_list)
    else:
        not_diff = [well_id_list]

#%%
#==================================================================
# Test 7: Compare the conductivity vs nitrate concentration.
#==================================================================

# Getting nuber of cases with correlation > corr_thresh_for_accept
cor_pos_test7, high_corr_test7 = corr_process(df,xvar = "mean_nitrate", yvar = "Conductivity", 
                min_sample_for_corr = min_sample_for_corr, corr_thresh_for_accept = corr_thresh_for_accept)
print(cor_pos_test7)

# Total highly correlated station lists
len(high_corr_test7)

#%%
# Plot scatter of conductivity vs nitrate concentration
scatter_twovars(df, selected_well_ids = not_diff[high_corr_test7[0]], xvar = "Conductivity", yvar ="mean_nitrate",
                    xlab = "Conductivity", ylab = "Nitrate concentration")

# %%
#==================================================================
# Test 8: Compare CAFO population vs nitrate.
#==================================================================
# Getting nuber of cases with correlation > corr_thresh_for_accept
cor_pos_test8, high_corr_test8 = corr_process(df,xvar = "mean_nitrate", yvar = "Cafo_Population_5miles", 
                min_sample_for_corr = min_sample_for_corr, corr_thresh_for_accept = corr_thresh_for_accept)
print(cor_pos_test8)

# Total highly correlated station lists
len(high_corr_test8)

# Plot scatter of CAFO population vs nitrate concentration
scatter_twovars(df, selected_well_ids = not_diff[high_corr_test8[0]], xvar = "Cafo_Population_5miles", yvar ="mean_nitrate",
                    xlab = "Animal population", ylab = "Nitrate concentration")

# %%
#==================================================================
# Test 9: Compare Agricultural area vs nitrate
#==================================================================
# Getting nuber of cases with correlation > corr_thresh_for_accept
cor_pos_test9, high_corr_test9 = corr_process(df,xvar = "Average_ag_area", yvar = "mean_nitrate", 
                min_sample_for_corr = min_sample_for_corr, corr_thresh_for_accept = corr_thresh_for_accept)
print(cor_pos_test9)

# Total highly correlated station lists
len(high_corr_test9)

# Plot scatter of CAFO population vs nitrate concentration
scatter_twovars(df, selected_well_ids = not_diff[high_corr_test9[0]], xvar = "Average_ag_area", yvar ="mean_nitrate",
                    xlab = "Animal population", ylab = "Nitrate concentration")

# %%
