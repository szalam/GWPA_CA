#%%
import sys
sys.path.insert(0,'src')
sys.path.insert(0,'src/data')
import pandas as pd
import config
import numpy as np
import matplotlib.pyplot as plt 
import ppfun as dp
import seaborn as sns
import matplotlib as mpl

df_main = pd.read_csv(config.data_processed / "Dataset_processed.csv")
#%%
df = df_main.copy()
lyrs = 9
rad_well = 2
cond_type_used = f'Resistivity_lyrs_{lyrs}' #'Conductivity_lyrs_9' # 'Conductivity_lyrs_1'
# cond_type_used = 'Conductivity_lyrs_9'
if cond_type_used == 'Conductivity_lyrs_9' or cond_type_used == 'Conductivity_lyrs_1':
    aem_type = 'Conductivity'
else:
    aem_type = 'Resistivity'

# Read dataset
df = df[df.well_data_source == 'GAMA']
# df = df[df.measurement_count > 4]
# df = df[df.city_inside_outside == 'outside_city']
# df = df[df.city_inside_outside == 'inside_city']
# df = df[[f'{cond_type_used}','mean_nitrate']]
# df = df[df['SubRegion'] != 14]
# df = df[df['SubRegion']==6]
# df = df[df[f'thickness_abovCond_{round(.1*100)}'] == 0 ]
# well_type_select = 'Domestic' # 'Water Supply, Other', 'Municipal', 'Domestic'
# df = df[df.well_type ==  well_type_select] 
# df = df[df.All_crop_2015 >= 2.030499e+08]

# Remove high salinity regions
exclude_subregions = [14, 15, 10, 19,18, 9,6]
# filter aemres to keep only the rows where Resistivity is >= 10 and SubRegion is not in the exclude_subregions list
df = df[(df[f'thickness_abovCond_{round(.1*100)}_lyrs_9_rad_{rad_well}miles'] <= 31) | (~df['SubRegion'].isin(exclude_subregions))]
# df = df[df.mean_nitrate>10]
# df = df.dropna()



# df = df[['well_id','Conductivity','mean_nitrate','area_wt_sagbi', 'Cafo_Population_5miles','Average_ag_area','change_per_year','total_ag_2010','APPROXIMATE LATITUDE', 'APPROXIMATE LONGITUDE','city_inside_outside']]
# df = df.head(1000)
#%%
df_exp = df[[f'{cond_type_used}','mean_nitrate']]

def calculate_scores(df, nitrate_threshold, resistivity_threshold_range):
    scores = []
    for threshold in resistivity_threshold_range:
        df_scores = add_columns(df, nitrate_threshold, threshold)
        total_score = df_scores['score'].sum()
        scores.append(total_score)
    return scores

def plot_scores(resistivity_threshold_range, scores):
    plt.bar(resistivity_threshold_range, scores)
    plt.xlabel('Resistivity Threshold')
    plt.ylabel('Total Score')
    # plt.ylim(0,10000)
    plt.show()

resistivity_threshold_range = range(10, 51) # resistivity thresholds from 10 to 50
scores = calculate_scores(df_exp, nitrate_threshold=50, resistivity_threshold_range=resistivity_threshold_range)
plot_scores(resistivity_threshold_range, scores)


# %%
#============================= Heatmap ==============================
df_exp = df[[f'{cond_type_used}','mean_nitrate']]

def calculate_scores(df, nitrate_threshold_range, resistivity_threshold_range):
    scores = []
    for nitrate_threshold in nitrate_threshold_range:
        row_scores = []
        for resistivity_threshold in resistivity_threshold_range:
            df_scores = add_columns(df, nitrate_threshold, resistivity_threshold)
            total_score = df_scores['score'].sum()
            row_scores.append(total_score)
        scores.append(row_scores)
    return scores

def plot_scores(scores, nitrate_threshold_range, resistivity_threshold_range):
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(scores)

    # set x-axis and y-axis labels and ticks
    ax.set_xticks(range(len(resistivity_threshold_range)))
    ax.set_yticks(range(len(nitrate_threshold_range)))
    ax.set_xticklabels(resistivity_threshold_range)
    ax.set_yticklabels(nitrate_threshold_range)
    ax.set_xlabel('Resistivity Threshold')
    ax.set_ylabel('Nitrate Threshold')

    # add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Total Score', rotation=-90, va="bottom")

    # set plot title
    ax.set_title('Score vs Nitrate and Resistivity Thresholds')

    # loop over data dimensions and create text annotations
    for i in range(len(nitrate_threshold_range)):
        for j in range(len(resistivity_threshold_range)):
            text = ax.text(j, i, scores[i][j], ha="center", va="center", color="w")

    # show plot
    plt.show()

# example usage
nitrate_threshold_range = range(10, 25,3) # nitrate thresholds from 5 to 50
resistivity_threshold_range = range(10, 50,5) # resistivity thresholds from 10 to 50

scores = calculate_scores(df_exp, nitrate_threshold_range, resistivity_threshold_range)
plot_scores(scores, nitrate_threshold_range, resistivity_threshold_range)
# %%
# example usage
nitrate_threshold_range = range(25, 50,3) # nitrate thresholds from 5 to 50
resistivity_threshold_range = range(10, 50,5) # resistivity thresholds from 10 to 50

scores = calculate_scores(df_exp, nitrate_threshold_range, resistivity_threshold_range)
plot_scores(scores, nitrate_threshold_range, resistivity_threshold_range)

#%%
def calculate_probabilities(df, nitrate_threshold_range, resistivity_threshold_range):
    probabilities = []
    for nitrate_threshold in nitrate_threshold_range:
        row_probs = []
        for resistivity_threshold in resistivity_threshold_range:
            df_scores = add_columns(df, nitrate_threshold, resistivity_threshold)
            num_samples = len(df_scores)
            num_samples_pass = df_scores['score'].sum()
            row_probs.append(num_samples_pass / num_samples)
        probabilities.append(row_probs)

    return probabilities

def plot_probabilities(probabilities, nitrate_threshold_range, resistivity_threshold_range):
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(probabilities, cmap='coolwarm')

    # set x-axis and y-axis labels and ticks
    ax.set_xticks(range(len(resistivity_threshold_range)))
    ax.set_yticks(range(len(nitrate_threshold_range)))
    ax.set_xticklabels(resistivity_threshold_range)
    ax.set_yticklabels(nitrate_threshold_range)
    ax.set_xlabel('Resistivity Threshold')
    ax.set_ylabel('Nitrate Threshold')

    # add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Probability of Pass', rotation=-90, va="bottom")

    # set plot title
    ax.set_title('Probability of Pass vs Nitrate and Resistivity Thresholds')

    # loop over data dimensions and create text annotations
    for i in range(len(nitrate_threshold_range)):
        for j in range(len(resistivity_threshold_range)):
            text = ax.text(j, i, '{:.2f}'.format(probabilities[i][j]), ha="center", va="center", color="w")

    # show plot
    plt.show()

# example usage
df_exp = df[[f'{cond_type_used}','mean_nitrate']]
nitrate_threshold_range = range(10, 25, 3) # nitrate thresholds from 10 to 22
resistivity_threshold_range = range(10, 50, 5) # resistivity thresholds from 10 to 45

probabilities = calculate_probabilities(df_exp, nitrate_threshold_range, resistivity_threshold_range)
plot_probabilities(probabilities, nitrate_threshold_range, resistivity_threshold_range)
# %%
