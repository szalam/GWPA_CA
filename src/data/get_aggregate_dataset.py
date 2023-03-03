#%%
import os
import sys
sys.path.insert(0,'src')
import config
import pandas as pd

# list of all csv files
csv_files_ucd = [f'{config.data_processed}/well_stats/ucdnitrate_stats.csv', 
            f'{config.data_processed}/gwdepth_wellbuff/GWDepth_wellsrc_UCD_rad_2mile.csv', 
            f'{config.data_processed}/sagbi_values/SAGBI_wellsrc_UCD_rad_2mile.csv', 
            f'{config.data_processed}/cafo_pop_wellbuffer/Cafopop_wellsrc_UCD_rad_2mile.csv', 
            f'{config.data_processed}/cafo_pop_wellbuffer/Cafopop_wellsrc_UCD_rad_5mile.csv', 
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_UCD_rad_2mile_lyrs_1.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_UCD_rad_2mile_lyrs_4.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_UCD_rad_2mile_lyrs_6.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_UCD_rad_2mile_lyrs_9.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_UCD_rad_2mile_layerwide.csv']

csv_files_gama = [f'{config.data_processed}/well_stats/gamanitrate_stats.csv', 
            f'{config.data_processed}/gwdepth_wellbuff/GWDepth_wellsrc_GAMA_rad_2mile.csv', 
            f'{config.data_processed}/sagbi_values/SAGBI_wellsrc_GAMA_rad_2mile.csv', 
            f'{config.data_processed}/cafo_pop_wellbuffer/Cafopop_wellsrc_GAMA_rad_2mile.csv',
            f'{config.data_processed}/cafo_pop_wellbuffer/Cafopop_wellsrc_GAMA_rad_5mile.csv', 
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_GAMA_rad_2mile_lyrs_1.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_GAMA_rad_2mile_lyrs_4.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_GAMA_rad_2mile_lyrs_6.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_GAMA_rad_2mile_lyrs_9.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_GAMA_rad_2mile_layerwide.csv']

def get_combined_dataset(csv_files):
    # read the first csv file
    df = pd.read_csv(csv_files[0])

    # rename column 'WELL ID' to 'well_id' if it exists
    if 'WELL ID' in df.columns:
        df.rename(columns={'WELL ID': 'well_id'}, inplace=True)

    # remove duplicate based on well_id
    df = df.drop_duplicates(subset='well_id', keep='first')

    # Iterate over the rest of the csv files
    for csv_file in csv_files[1:]:
        df2 = pd.read_csv(csv_file)
        # rename column 'WELL ID' to 'well_id' if it exists
        if 'WELL ID' in df2.columns:
            df2.rename(columns={'WELL ID': 'well_id'}, inplace=True)

        # remove duplicate based on well_id
        df2 = df2.drop_duplicates(subset='well_id', keep='first')

        # merge the dataframe with the current csv file based on well_id
        df = pd.merge(df, df2, on='well_id', how='outer')

    return(df)

df_ucd = get_combined_dataset(csv_files_ucd)
df_ucd['well_data_source'] = 'UCD'
df_gama = get_combined_dataset(csv_files_gama)
df_gama['well_data_source'] = 'GAMA'

# Combine two dataset
df = pd.concat([df_ucd, df_gama], axis=0)

columns_to_keep = ['well_id', 'APPROXIMATE LATITUDE',
       'APPROXIMATE LONGITUDE', 'mean_nitrate', 'median_nitrate',
       'max_nitrate', 'min_nitrate', 'measurement_count',
       'mean_concentration_2015-2022', 'mean_concentration_2010-2015',
       'mean_concentration_2005-2010', 'mean_concentration_2000-2005',
       'mean_concentration_2000-2022', 'mean_concentration_2010-2022',
       'mean_concentration_2007-2009', 'mean_concentration_2012-2015',
       'mean_concentration_2019-2021', 'mean_concentration_2017-2018',
       'trend','change_per_year', 'start_date', 'end_date', 
       'gwdep', 'area_wt_sagbi', 'total_obs', 'well_type',
       'CAFO_Population_2miles','CAFO_Population_5miles', 'Conductivity_lyrs_9',
       'Conductivity_lyrs_6','Conductivity_lyrs_4','Conductivity_lyrs_1','well_data_source']

for i in range(1,21):
    column_name = f'Conductivity_depthwtd_lyr{i}'
    columns_to_keep.append(column_name)

df = df[columns_to_keep]

# Inverse conductivity to get depth average resistivity
df['Resistivity_lyrs_9'] = 1/df['Conductivity_lyrs_9']
df['Resistivity_lyrs_6'] = 1/df['Conductivity_lyrs_6']
df['Resistivity_lyrs_4'] = 1/df['Conductivity_lyrs_4']
df['Resistivity_lyrs_1'] = 1/df['Conductivity_lyrs_1']
#%%

# df = df[['well_id', 'APPROXIMATE LATITUDE',
#        'APPROXIMATE LONGITUDE', 'mean_nitrate', 'median_nitrate',
#        'max_nitrate', 'min_nitrate', 'measurement_count',
#        'mean_concentration_2015-2022', 'mean_concentration_2010-2015',
#        'mean_concentration_2005-2010', 'mean_concentration_2000-2005',
#        'mean_concentration_2000-2022', 'mean_concentration_2010-2022',
#        'mean_concentration_2007-2009', 'mean_concentration_2012-2015',
#        'mean_concentration_2019-2021', 'mean_concentration_2017-2018',
#        'trend','change_per_year', 'start_date', 'end_date', 
#        'gwdep', 'area_wt_sagbi',
#        'Cafo_Population_2miles','Cafo_Population_5miles', 'Conductivity','well_data_source']]


#%%
# Check for duplicates in the 'well_id' column
duplicates = df[df.duplicated(['well_id'])]['well_id']

# Print the duplicate values
if len(duplicates) > 0:
    print("Duplicate values in 'well_id' column:")
    # print(duplicates.tolist())
else:
    print("No duplicate values in 'well_id' column.")



#%%
# Preprocessing the land use data
# Define the path to the UCD folder
ucd_folder = f'{config.data_processed}/CDL/cdl_at_buffers/UCD/'
gama_folder = f'{config.data_processed}/CDL/cdl_at_buffers/GAMA/'

# Create an empty list to store the dataframes
ucd_dfs = []
gama_dfs = []

# Iterate through the files in the UCD folder
for filename in os.listdir(ucd_folder):
    # Check if the file is a CSV
    if filename.endswith('.csv'):
        # Read the CSV into a dataframe
        df_cdl = pd.read_csv(ucd_folder + filename)
        df_cdl = df_cdl.drop(columns=df_cdl.columns[0])
        
        year = filename.split(".")[0].split("_")[1]
        
        # Adding all ag area
        df_cdl['total_ag'] = df_cdl[['Row_crops', 'Small_grains', 'Truck_nursery_berry', 'Citrus_subtropical', 'Field_crops', 'Vineyards', 'Grain_hay', 'Deciduous_fruits_nuts', 'Rice', 'Cotton', 'Other_crops']].sum(axis=1)

        # Rename the columns of the dataframe
        df_cdl.columns = [col + '_'+ year if col != 'well_id' else col for col in df_cdl.columns]
        # Append the dataframe to the list
        ucd_dfs.append(df_cdl)
#%%
# Iterate through the files in the GAMA folder
for filename in os.listdir(gama_folder):
    # Check if the file is a CSV
    if filename.endswith('.csv'):
        # Read the CSV into a dataframe
        df_cdl = pd.read_csv(gama_folder + filename)
        df_cdl = df_cdl.drop(columns=df_cdl.columns[0])
        # Adding all ag area
        df_cdl['total_ag'] = df_cdl[['Row_crops', 'Small_grains', 'Truck_nursery_berry', 'Citrus_subtropical', 'Field_crops', 'Vineyards', 'Grain_hay', 'Deciduous_fruits_nuts', 'Rice', 'Cotton', 'Other_crops']].sum(axis=1)
        
        year = filename.split(".")[0].split("_")[1]
        # Rename the columns of the dataframe
        df_cdl.columns = [col + '_'+ year if col != 'well_id' else col for col in df_cdl.columns]
        # Append the dataframe to the list
        gama_dfs.append(df_cdl)

#%%
# Merging the list items, which are dataframes
def merge_list_dfs(list_dfs = ucd_dfs):
    merged_df = list_dfs[0]

    for i in range(1,len(ucd_dfs)):
        merged_df = pd.merge(merged_df, list_dfs[i], on='well_id')
    return merged_df

# Concatenate all the dataframes in the list on the 'well_id' column
ucd_final_df = merge_list_dfs(list_dfs = ucd_dfs)
gama_final_df = merge_list_dfs(list_dfs = gama_dfs)
#%%
print(f'Shape if ucd_final df: {ucd_final_df.shape[0]}, \nwhile total unique wells: {len(ucd_final_df.well_id.unique())}')


#%%
df_cdl_all = pd.concat([ucd_final_df, gama_final_df], axis=0)

df_cdl_all['Average_ag_area'] = df_cdl_all[['total_ag_2007','total_ag_2008','total_ag_2009','total_ag_2010','total_ag_2011','total_ag_2012','total_ag_2013','total_ag_2014','total_ag_2015','total_ag_2016','total_ag_2017','total_ag_2018','total_ag_2019','total_ag_2020','total_ag_2021']].mean(axis=1)
#%%
# Fill all NaN with 0
df_cdl_all = df_cdl_all.fillna(0)

# merge the dataframe with the current csv file based on well_id
df = pd.merge(df, df_cdl_all, on='well_id', how='outer')

# Merge city in/out information
df_city_inout = pd.read_csv(config.data_processed / "well_inout_city/well_inout_city.csv")
#%%
# merge the dataframe with the current csv file based on well_id
df = pd.merge(df, df_city_inout, on='well_id', how='outer')

# Merge subregion information information
df_subreg = pd.read_csv(config.data_processed / "well_in_subregions.csv")
# merge the dataframe with the current csv file based on well_id
df = pd.merge(df, df_subreg, on='well_id', how='outer')

# Merge gwpa information
df_gwpa = pd.read_csv(config.data_processed / "well_in_gwpa.csv")
# merge the dataframe with the current csv file based on well_id
df = pd.merge(df, df_gwpa, on='well_id', how='outer')

# Read and merge the thickness for different conductivity threshold

def get_thickness_data(df, aem_lyr_lim = 4, dtype_sel = 'GAMA',rad_buffer = 2):
    # Set the values of cond_thresh to run the code for
    cond_thresh_values = [0.05, 0.08, 0.10, 0.15]
    # Loop over the cond_thresh values and run the code for each value
    for cond_thresh in cond_thresh_values:
        # Read the DataFrame from the CSV file
        df_thickness = pd.read_csv(config.data_processed / f"aem_values/Thickness_abovThresh_DWR_wellsrc_{dtype_sel}_rad_{rad_buffer}mile_lyrs_{aem_lyr_lim}_condThresh_{round(cond_thresh*100)}.csv")
        # Check if the column exists
        old_col_name = 'Conductivity_lyrs_9'
        old_col_name2 = f'thickness_abovCond_{round(cond_thresh*100)}'
        new_col_name = f'thickness_abovCond_{round(cond_thresh*100)}_lyrs_{aem_lyr_lim}_rad_{rad_buffer}miles'
        if old_col_name in df_thickness.columns:
            # Rename the column to the new name
            df_thickness = df_thickness.rename(columns={old_col_name: new_col_name})
        if old_col_name2 in df_thickness.columns:
            # Rename the column to the new name
            df_thickness = df_thickness.rename(columns={old_col_name2: new_col_name})

        # Select the columns we want to keep
        df_thickness = df_thickness[['well_id', new_col_name]]

        # Merge with the main DataFrame
        df = pd.merge(df, df_thickness, on='well_id', how='outer')

    return df

# df_tmp = df[['well_id']]
# df_gama_thickness_4 = get_thickness_data(df = df_tmp,aem_lyr_lim = 4, dtype_sel = 'GAMA',rad_buffer = 2)
# df_gama_thickness_6 = get_thickness_data(df = df_tmp,aem_lyr_lim = 6, dtype_sel = 'GAMA')
# df_gama_thickness_9 = get_thickness_data(df = df_tmp,aem_lyr_lim = 9, dtype_sel = 'GAMA')
# df_ucd_thickness_4 = get_thickness_data(df = df_tmp,aem_lyr_lim = 4, dtype_sel = 'UCD')
# df_ucd_thickness_6 = get_thickness_data(df = df_tmp,aem_lyr_lim = 6, dtype_sel = 'UCD')
# df_ucd_thickness_9 = get_thickness_data(df = df_tmp,aem_lyr_lim = 9, dtype_sel = 'UCD')
# # Combine two dataset
# df_thickness = pd.concat([df_gama_thickness_4, df_gama_thickness_6,df_gama_thickness_9,
#                           df_ucd_thickness_4,df_ucd_thickness_6,df_ucd_thickness_9], axis=0)

df_tmp = df[['well_id']]
df_thickness_list = []

for aem_lyr_lim in [4, 6, 9]:
    for rad_buffer in [.5, 1, 2, 3, 4, 5]:
        df_gama_thickness = get_thickness_data(df=df_tmp, aem_lyr_lim=aem_lyr_lim, dtype_sel='GAMA', rad_buffer=rad_buffer)
        df_thickness_list.append(df_gama_thickness)

df_thickness = pd.concat(df_thickness_list)
df = pd.merge(df, df_thickness, on='well_id', how='outer')
#%%
# Use the str.replace() method to remove 'NO3_' from all values in the 'well_id' column
df['well_id'] = df['well_id'].str.replace('NO3_', '')

# export final dataframe to csv
df.to_csv(config.data_processed / "Dataset_processed.csv", index=False)

# %%