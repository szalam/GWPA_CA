#================================================================================
# This script analyzies the relationship between animal population and nitrate
# for wells when they have almost similar conductivity
#================================================================================
#%%
import sys
sys.path.insert(0,'src')

import pandas as pd
import config
import matplotlib.pyplot as plt 
from tqdm import tqdm

from scipy.stats import ttest_ind
from visualize import vismod

# Read dataset
df = pd.read_csv(config.data_processed / "Dataset_processed.csv")
df = df[df.well_data_source == 'GAMA']
df = df[['well_id','Conductivity','mean_nitrate','area_wt_sagbi', 'Cafo_Population','mean_concentration_2019-2021']]
# df = df.head(1000)

#get the unique well ids
well_ids = df["well_id"].unique()

#create an empty list to store the well ids with similar conductivity
not_diff = []

#iterate over the well ids
for well_id in tqdm(well_ids):
    # get the conductivity of the current well
    cond = df.query("well_id == @well_id")["Conductivity"].mean()

    # select all the rows that have conductivity within +-0.01 of the current well ID's conductivity
    data = df.query("well_id != @well_id and @cond - 0.00009 <= Conductivity <= @cond + 0.00005")
    well_id_list = list(data["well_id"].unique())

    if not_diff:
            not_diff.append(well_id_list)
    else:
        not_diff = [well_id_list]

#%%
# select a list item in not_diff that has well_id
selected_well_ids = not_diff[1000]

# separate data from df using those well_id
selected_data = df[df["well_id"].isin(selected_well_ids)]

# scatterplot the nitrate and animal polulation for those well_ids
plt.scatter(selected_data["Cafo_Population"], selected_data["mean_concentration_2019-2021"])
plt.xlabel("Animal population")
plt.ylabel("Nitrate concentration")
plt.show()


# %%
# Loop through all wells with similar conductivity values to see if there 
# is a posirivity correlation between nitrate and cafo values
cor_pos = 0
for i in tqdm(range(len(not_diff))):
    selected_well_ids = not_diff[i]

    # calculate the correlation between nitrate and animal population
    correlation = selected_data["mean_concentration_2010-2022"].corr(selected_data["Cafo_Population"])

    # check if the correlation is positive
    if correlation > .1:
        cor_pos += 1

print(cor_pos)

# Most correlation balues are between 0.1 and 0.2.
# %%
