#%%
import sys
sys.path.insert(0,'src')
import config
import pandas as pd

# list of all csv files
csv_files_ucd = [f'{config.data_processed}/well_stats/ucdnitrate_stats.csv', 
            f'{config.data_processed}/gwdepth_wellbuff/GWDepth_wellsrc_UCD_rad_2mile.csv', 
            f'{config.data_processed}/sagbi_values/SAGBI_wellsrc_UCD_rad_2mile.csv', 
            f'{config.data_processed}/cafo_pop_wellbuffer/Cafopop_wellsrc_UCD_rad_2mile.csv', 
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_UCD_rad_2mile.csv']

csv_files_gama = [f'{config.data_processed}/well_stats/gamanitrate_stats.csv', 
            f'{config.data_processed}/gwdepth_wellbuff/GWDepth_wellsrc_GAMA_rad_2mile.csv', 
            f'{config.data_processed}/sagbi_values/SAGBI_wellsrc_GAMA_rad_2mile.csv', 
            f'{config.data_processed}/cafo_pop_wellbuffer/Cafopop_wellsrc_GAMA_rad_2mile.csv', 
            f'{config.data_processed}/aem_values/AEMsrc_DWR_wellsrc_GAMA_rad_2mile.csv']

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


df = df[['well_id', 'APPROXIMATE LATITUDE',
       'APPROXIMATE LONGITUDE', 'mean_nitrate', 'median_nitrate',
       'max_nitrate', 'min_nitrate', 'measurement_count',
       'mean_concentration_2015-2022', 'mean_concentration_2010-2015',
       'mean_concentration_2005-2010', 'mean_concentration_2000-2005',
       'mean_concentration_2000-2022', 'mean_concentration_2010-2022',
       'mean_concentration_2007-2009', 'mean_concentration_2012-2015',
       'mean_concentration_2019-2021', 'mean_concentration_2017-2018',
       'start_date', 'end_date', 'gwdep', 'area_wt_sagbi',
       'Cafo_Population', 'Conductivity','well_data_source']]

# export final dataframe to csv
df.to_csv(config.data_processed / "Dataset_processed.csv", index=False)

# %%
