#%%
import os
import sys
sys.path.insert(0,'src')
import config
import pandas as pd
import numpy as np

# list of all csv files
csv_files_ucd = [f'{config.data_processed}/well_stats/ucdnitrate_stats.csv', 
            f'{config.data_processed}/gwdepth_wellbuff/GWDepth_wellsrc_UCD_rad_2mile.csv', 
            f'{config.data_processed}/sagbi_values/SAGBI_wellsrc_UCD_rad_2mile.csv', 
            f'{config.data_processed}/cafo_pop_wellbuffer/Cafopop_wellsrc_UCD_rad_2mile.csv', 
            f'{config.data_processed}/cafo_pop_wellbuffer/Cafopop_wellsrc_UCD_rad_5mile.csv', 
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_UCD_rad_2mile_lyrs_1.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_UCD_rad_2mile_lyrs_4.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_UCD_rad_2mile_lyrs_6.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_UCD_rad_0.2mile_lyrs_9.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_UCD_rad_0.5mile_lyrs_9.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_UCD_rad_1mile_lyrs_9.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_UCD_rad_1.5mile_lyrs_9.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_UCD_rad_2mile_lyrs_9.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_UCD_rad_2.5mile_lyrs_9.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_UCD_rad_3mile_lyrs_9.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_UCD_rad_3.5mile_lyrs_9.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_UCD_rad_2mile_layerwide.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_UCD_rad_1mile_layerwide.csv']

csv_files_gama = [f'{config.data_processed}/well_stats/gamanitrate_stats.csv', 
            f'{config.data_processed}/gwdepth_wellbuff/GWDepth_wellsrc_GAMA_rad_2mile.csv', 
            f'{config.data_processed}/sagbi_values/SAGBI_wellsrc_GAMA_rad_2mile.csv', 
            f'{config.data_processed}/cafo_pop_wellbuffer/Cafopop_wellsrc_GAMA_rad_2mile.csv',
            f'{config.data_processed}/cafo_pop_wellbuffer/Cafopop_wellsrc_GAMA_rad_5mile.csv', 
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_GAMA_rad_2mile_lyrs_1.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_GAMA_rad_2mile_lyrs_4.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_GAMA_rad_2mile_lyrs_6.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_GAMA_rad_0.2mile_lyrs_9.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_GAMA_rad_0.5mile_lyrs_9.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_GAMA_rad_1mile_lyrs_9.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_GAMA_rad_1.5mile_lyrs_9.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_GAMA_rad_2mile_lyrs_9.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_GAMA_rad_2.5mile_lyrs_9.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_GAMA_rad_3mile_lyrs_9.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_GAMA_rad_3.5mile_lyrs_9.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_GAMA_rad_2mile_layerwide.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_GAMA_rad_1mile_layerwide.csv']

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

#%%
# Combine two dataset
df = pd.concat([df_ucd, df_gama], axis=0)
# df = df_gama

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
       'CAFO_Population_2miles','CAFO_Population_5miles', 
       'Conductivity_lyrs_9_rad_0.2mile','Conductivity_lyrs_9_rad_0.5mile','Conductivity_lyrs_9_rad_1mile','Conductivity_lyrs_9_rad_1.5mile','Conductivity_lyrs_9_rad_2mile','Conductivity_lyrs_9_rad_2.5mile','Conductivity_lyrs_9_rad_3mile','Conductivity_lyrs_9_rad_3.5mile',
       'Conductivity_lyrs_6','Conductivity_lyrs_4','Conductivity_lyrs_1','well_data_source']

for i in range(2,21):
    column_name = f'Conductivity_depthwtd_lyr{i}_rad_2mile'
    column_name2 = f'Conductivity_depthwtd_lyr{i}_rad_1mile'
    columns_to_keep.append(column_name)
    columns_to_keep.append(column_name2)

df = df[columns_to_keep]
#%%
# Inverse conductivity to get depth average resistivity
df['Resistivity_lyrs_9_rad_0_2_miles'] = 1/df['Conductivity_lyrs_9_rad_0.2mile']
df['Resistivity_lyrs_9_rad_0_5_miles'] = 1/df['Conductivity_lyrs_9_rad_0.5mile']
df['Resistivity_lyrs_9_rad_1_miles'] = 1/df['Conductivity_lyrs_9_rad_1mile']
df['Resistivity_lyrs_9_rad_1_5_miles'] = 1/df['Conductivity_lyrs_9_rad_1.5mile']
df['Resistivity_lyrs_9_rad_2_miles'] = 1/df['Conductivity_lyrs_9_rad_2mile']
df['Resistivity_lyrs_9_rad_2_5_miles'] = 1/df['Conductivity_lyrs_9_rad_2.5mile']
df['Resistivity_lyrs_9_rad_3_miles'] = 1/df['Conductivity_lyrs_9_rad_3mile']
df['Resistivity_lyrs_9_rad_3_5_miles'] = 1/df['Conductivity_lyrs_9_rad_3.5mile']
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

# merge the dataframe with the current csv file based on well_id
df = pd.merge(df, df_city_inout, on='well_id', how='left')
#%%
# Merge subregion information information
df_subreg = pd.read_csv(config.data_processed / "well_in_subregions.csv")

# Drop duplicates based on well_id in the df_subreg DataFrame
df_subreg = df_subreg.drop_duplicates(subset='well_id')
# merge the dataframe with the current csv file based on well_id
df = pd.merge(df, df_subreg, on='well_id', how='left')
#%%
# Merge gwpa information
df_gwpa = pd.read_csv(config.data_processed / "well_in_gwpa.csv")

# Drop duplicates based on well_id in the df_subreg DataFrame
df_gwpa = df_gwpa.drop_duplicates(subset='well_id')
# merge the dataframe with the current csv file based on well_id
df = pd.merge(df, df_gwpa, on='well_id', how='left')

#%%
# df = df2.copy()
df_thickness = pd.read_csv(config.data_processed / 'aem_values/thickness_combined_rad_cond_lyrs.csv')
df = pd.merge(df, df_thickness, on='well_id', how='left')

#%% import the redox condition
# List of file names
file_redox = ["ProbDOpt5ppm_Deep", "ProbDOpt5ppm_Shallow", "ProbMn50ppb_Deep", "ProbMn50ppb_Shallow"]

# Dictionary to store the dataframes
df_dict = {}

# Load each file into a separate dataframe
for file_name in file_redox:
    df_tmp = pd.read_csv(config.data_processed / 'redox_Ninput_katetal/exported_csv_redox_Ninput' / f"{file_name}.csv")
    df_tmp['mean_value'] = df_tmp['mean_value'].apply(lambda x: np.nan if x < 0 else x)
    df_tmp = df_tmp.rename(columns={"mean_value": file_name})
    # Replace negative values with NaN
    df_dict[file_name] = df_tmp

# Merge the dataframes based on well_id
redox_df = df_dict[file_redox[0]]
for file_name in file_redox[1:]:
    redox_df = pd.merge(redox_df, df_dict[file_name], on="well_id", how="outer")

# Keep only the well_id and columns with names in file_names
columns_to_keep = ["well_id"] + file_redox
redox_df = redox_df[columns_to_keep]

df = pd.merge(df, redox_df, on='well_id', how='left')

#%%
# # Import N input data
# df_N_input = pd.read_csv(config.data_processed / 'redox_Ninput_katetal/exported_csv_redox_Ninput' / 'N_total.csv')
# df_N_input = df_N_input.rename(columns={"mean_value": 'N_total'})
# df_N_input = df_N_input[['well_id','N_total']]
# # Replace negative values with NaN
# df_N_input['N_total'] = df_N_input['N_total'].apply(lambda x: np.nan if x < 0 else x)

# # merge the dataframe with the current csv file based on well_id
# df = pd.merge(df, df_N_input, on='well_id', how='left')

def process_and_merge_data(df, file_name, column_name):
    df_input = pd.read_csv(config.data_processed / f'redox_Ninput_katetal/exported_csv_redox_Ninput/{file_name}.csv')
    df_input = df_input.rename(columns={"mean_value": column_name})
    df_input = df_input[['well_id', column_name]]
    # Replace negative values with NaN
    df_input[column_name] = df_input[column_name].apply(lambda x: np.nan if x < 0 else x)
    # merge the dataframe with the current csv file based on well_id
    df = pd.merge(df, df_input, on='well_id', how='left')
    return df

# Assuming df is already defined
variables = [
    'CAML1990_natural_water',
    'CVHM_TextZone',
    'DTW60YrJurgens',
    'HiWatTabDepMin',
    'LateralPosition',
    'Ngw_1975',
    'PrecipMinusETin_1971_2000_GWRP',
    'RechargeAnnualmmWolock',
    'RiverDist_NEAR',
    'ScreenLength_Deep',
    'ScreenLength_Shallow',
    'N_total'
]

for var in variables:
    df = process_and_merge_data(df, var, var)

#%%
# Use the str.replace() method to remove 'NO3_' from all values in the 'well_id' column
df['well_id'] = df['well_id'].str.replace('NO3_', '')
#%%
# export final dataframe to csv
df.to_csv(config.data_processed / "Dataset_processed.csv", index=False)

# %%
def check_duplicates(df):
    # Check for duplicate 'WELL ID's in the DataFrame
    duplicate_well_ids = df[df.duplicated(subset='well_id', keep=False)]

    # Count the number of duplicate well IDs
    duplicate_well_id_count = len(duplicate_well_ids)

    print(f"Number of duplicate well IDs: {duplicate_well_id_count}")
# %%
