#%%
import os
import sys
sys.path.insert(0,'src')
import config
import pandas as pd
import numpy as np

#%%
csv_files_gama = [f'{config.data_processed}/well_stats/gamanitrate_latest_stats.csv', 
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_GAMAlatest_rad_0.5mile_lyrs_9.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_GAMAlatest_rad_1mile_lyrs_9.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_GAMAlatest_rad_1.5mile_lyrs_9.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_GAMAlatest_rad_2mile_lyrs_9.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_GAMAlatest_rad_2.5mile_lyrs_9.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_GAMAlatest_rad_3mile_lyrs_9.csv',
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_GAMAlatest_rad_3.5mile_lyrs_9.csv']

def get_combined_dataset(csv_files):
    # read the first csv file
    df = pd.read_csv(csv_files[0])

    # rename column 'WELL ID' to 'well_id' if it exists
    if 'WELL ID' in df.columns:
        df.rename(columns={'WELL ID': 'well_id'}, inplace=True)

    # remove duplicate based on well_id
    df = df.drop_duplicates(subset='well_id', keep='first')

    # drop 'Unnamed: 0' if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')

    # Iterate over the rest of the csv files
    for csv_file in csv_files[1:]:
        df2 = pd.read_csv(csv_file)

        # rename column 'WELL ID' to 'well_id' if it exists
        if 'WELL ID' in df2.columns:
            df2.rename(columns={'WELL ID': 'well_id'}, inplace=True)

        # remove duplicate based on well_id
        df2 = df2.drop_duplicates(subset='well_id', keep='first')

        # drop 'Unnamed: 0' if it exists
        if 'Unnamed: 0' in df2.columns:
            df2 = df2.drop(columns='Unnamed: 0')

        # exclude columns that are already in the master df
        df2 = df2.drop(columns=[col for col in df2.columns if col in df.columns and col != 'well_id'])

        # merge the dataframe with the current csv file based on well_id
        df = pd.merge(df, df2, on='well_id', how='outer')

    return df


df_gama = get_combined_dataset(csv_files_gama)
df_gama['well_data_source'] = 'GAMA'

#%%
# Check for duplicates in df
print(df_gama.duplicated('well_id').any())

#%%
# Combine two dataset
df = df_gama.copy()
# df = df_gama

columns_to_keep = ['well_id', 'APPROXIMATE LATITUDE',
       'APPROXIMATE LONGITUDE', 'mean_nitrate', 'median_nitrate',
       'max_nitrate', 'min_nitrate', 'measurement_count',
    #    'trend','change_per_year', 
       'start_date', 'end_date', 
       'total_obs', 'well_type',
       'Conductivity_lyrs_9_rad_0.5mile','Conductivity_lyrs_9_rad_1mile','Conductivity_lyrs_9_rad_1.5mile','Conductivity_lyrs_9_rad_2mile','Conductivity_lyrs_9_rad_2.5mile','Conductivity_lyrs_9_rad_3mile','Conductivity_lyrs_9_rad_3.5mile',
       'well_data_source']

# Start with the periods you explicitly defined
explicit_periods = [
    "2015-2022", "2010-2015", "2005-2010", "2000-2005", 
    "2000-2022", "2010-2022", "2007-2009", "2012-2015", 
    "2019-2021", "2017-2018"
]

for period in explicit_periods:
    columns_to_keep.append(f'mean_concentration_{period}')

# Now add columns for every 3-year period from 1990 to 2022
for year in range(1990, 2022, 3):
    key = f"{year}-{year + 2}" # Adjusted the end year to make it a 3-year period without overlap
    if key not in explicit_periods:
        columns_to_keep.append(f'mean_concentration_{key}')

# And add columns for every 5-year period from 1990 to 2022
for year in range(1990, 2022, 5):
    key = f"{year}-{year + 4}" # Adjusted the end year to make it a 5-year period without overlap
    if key not in explicit_periods:
        columns_to_keep.append(f'mean_concentration_{key}')

#%%
# Inverse conductivity to get depth average resistivity
df['Resistivity_lyrs_9_rad_0_5_miles'] = 1/df['Conductivity_lyrs_9_rad_0.5mile']
df['Resistivity_lyrs_9_rad_1_miles'] = 1/df['Conductivity_lyrs_9_rad_1mile']
df['Resistivity_lyrs_9_rad_1_5_miles'] = 1/df['Conductivity_lyrs_9_rad_1.5mile']
df['Resistivity_lyrs_9_rad_2_miles'] = 1/df['Conductivity_lyrs_9_rad_2mile']
df['Resistivity_lyrs_9_rad_2_5_miles'] = 1/df['Conductivity_lyrs_9_rad_2.5mile']
df['Resistivity_lyrs_9_rad_3_miles'] = 1/df['Conductivity_lyrs_9_rad_3mile']
df['Resistivity_lyrs_9_rad_3_5_miles'] = 1/df['Conductivity_lyrs_9_rad_3.5mile']

#%%
# Check for duplicates in df
print(df.duplicated('well_id').any())

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
df_thickness = pd.read_csv(config.data_processed / 'aem_values/thickness_combined_rad_cond_lyrs_GAMAlatest.csv')
df = pd.merge(df, df_thickness, on='well_id', how='left')

#%%
# Check for duplicates in df
print(df.duplicated('well_id').any())

#%% import the redox condition
# List of file names
file_redox = ["ProbDOpt5ppm_Deep", "ProbDOpt5ppm_Shallow", "ProbMn50ppb_Deep", "ProbMn50ppb_Shallow"]

# Dictionary to store the dataframes
df_dict = {}

# Load each file into a separate dataframe
for file_name in file_redox:
    df_tmp = pd.read_csv(config.data_processed / 'redox_Ninput_katetal/exported_csv_redox_Ninput' / f"{file_name}_GAMAlatest_rad_2mil.csv")
    df_tmp['mean_value'] = df_tmp['mean_value'].apply(lambda x: np.nan if x < 0 else x)
    df_tmp = df_tmp.rename(columns={"mean_value": file_name})

    # drop 'Unnamed: 0' if it exists
    if 'Unnamed: 0' in df_tmp.columns:
        df_tmp = df_tmp.drop(columns='Unnamed: 0')

    # Replace negative values with NaN
    df_dict[file_name] = df_tmp

# Merge the dataframes based on well_id
redox_df = df_dict[file_redox[0]]
for file_name in file_redox[1:]:
    redox_df = pd.merge(redox_df, df_dict[file_name], on="well_id", how="outer")

# Keep only the well_id and columns with names in file_names
columns_to_keep = ["well_id"] + file_redox
redox_df = redox_df[columns_to_keep]

# Handle duplicates: group by 'well_id' and take the mean of other columns
redox_df = redox_df.groupby('well_id').mean().reset_index()

#%%
# Check for duplicates in df
print(redox_df.duplicated('well_id').any())
#%%
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

def process_and_merge_data(df, file_name, column_name, rad = 2):
    df_input = pd.read_csv(config.data_processed / f'redox_Ninput_katetal/exported_csv_redox_Ninput/{file_name}_GAMAlatest_rad_{rad}mil.csv')
    df_input = df_input.rename(columns={"mean_value": column_name})
    df_input = df_input[['well_id', column_name]]
    # Handle duplicates: group by 'well_id' and take the mean of other columns
    df_input = df_input.groupby('well_id').mean().reset_index()

    # Replace negative values with NaN
    # df_input[column_name] = df_input[column_name].apply(lambda x: np.nan if x < 0 else x)
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
    df = process_and_merge_data(df, var, var, rad=2)

#%%
# Check for duplicates in df
print(df.duplicated('well_id').any())

#%%
# Use the str.replace() method to remove 'NO3_' from all values in the 'well_id' column
df['well_id'] = df['well_id'].str.replace('NO3_', '')
#%%
# export final dataframe to csv
df.to_csv(config.data_processed / "Dataset_processed_GAMAlatest.csv", index=False)

# %%
def check_duplicates(df):
    # Check for duplicate 'WELL ID's in the DataFrame
    duplicate_well_ids = df[df.duplicated(subset='well_id', keep=False)]

    # Count the number of duplicate well IDs
    duplicate_well_id_count = len(duplicate_well_ids)

    print(f"Number of duplicate well IDs: {duplicate_well_id_count}")
check_duplicates(df)
# %%
