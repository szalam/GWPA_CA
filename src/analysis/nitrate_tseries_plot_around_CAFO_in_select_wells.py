
#%%
import sys
sys.path.insert(0,'src')
sys.path.insert(0,'src/data')

import pandas as pd
import config
import matplotlib.pyplot as plt 
import ppfun as dp

# Read dataset
df = pd.read_csv(config.data_processed / "Dataset_processed.csv")
df['nitrate_increase'] = df['mean_concentration_2015-2022']- df['mean_concentration_2005-2010']

# Separate data from GAMA and UCD
dfgama = df[df.well_data_source == 'GAMA']
dfucd = df[df.well_data_source == 'UCD']

# %%
well_src = 'UCD'
#=========================== Import water quality data ==========================
if well_src == 'GAMA':
    # read gama excel file
    df = pd.read_excel(config.data_gama / 'TULARE_NO3N.xlsx',engine='openpyxl')
    df.rename(columns = {'GM_WELL_ID':'well_id', 'GM_LATITUDE':'APPROXIMATE LATITUDE', 'GM_LONGITUDE':'APPROXIMATE LONGITUDE', 'GM_CHEMICAL_VVL': 'CHEMICAL', 'GM_RESULT': 'RESULT','GM_WELL_CATEGORY':'DATASET_CAT','GM_SAMP_COLLECTION_DATE':'DATE'}, inplace = True)
    df['DATE']= pd.to_datetime(df['DATE'])

if well_src == 'UCD':
    # file location
    file_polut = config.data_raw / "nitrate_data/UCDNitrateData.csv"

    # Read nitrate data
    df = dp.get_polut_df(file_sel = file_polut)
    df.rename(columns = {'WELL ID':'well_id'}, inplace = True)
    


#%%
#===================================================================================================================
# Plot time series of nitrate of wells having positive correlation between nitrate and CAFO but close conductivity
#===================================================================================================================

def plot_time_series(df, ucd_ids, plt_row = 7, plt_colm = 4):
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
            well_id = f'NO3_{ucd_id}'
            dp.get_plot_time_series_well_canvas(df, well_id = well_id, xvar = 'DATE', yvar = 'RESULT', ax=ax[plot_num])
        plt.tight_layout()
        plt.show()

# Read the csv file
df_well_cafo_pos = pd.read_csv(config.data_processed / "cafo_N_positive_relation_same_conductivity/wellids_have_cafo_positive_relations.csv")

# Get the list of ucd_ids from the csv file
ucd_ids = df_well_cafo_pos['well_id'].tolist()

# Plot the time series for each ucd_id
plot_time_series(df, ucd_ids,plt_row = 6, plt_colm = 5)
# %%
