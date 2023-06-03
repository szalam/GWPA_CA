#%%
# Import the necessary libraries
import os
import sys
import pandas as pd
import sys
sys.path.insert(0, 'src') # Add 'src' directory to the system path
import config
from scipy.stats import linregress
from scipy import stats

# Set the path to the directory containing the CSV files
path = f'{config.data_raw}/GAMA/GAMAlatest'

# Get a list of all CSV files in the folder
csv_files = [file for file in os.listdir(path) if file.endswith('.csv')]

# Initialize an empty DataFrame to store the combined data
combined_df = pd.DataFrame()

# Read and combine the CSV files
for file in csv_files:
    print(f'Reading {file}')
    file_path = os.path.join(path, file)
    df = pd.read_csv(file_path, encoding='latin1')
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# Display the first few rows of the combined DataFrame
print(combined_df.head())

# Select rows where the 'gm_chemical_vvl' column is 'NO3N'
df_nitrate_main = combined_df[combined_df.gm_chemical_vvl == 'NO3N']
print("Shape of df_nitrate_main:", df_nitrate_main.shape)

# Print the column names of the selected DataFrame
print("Column names of df_nitrate_main:", df_nitrate_main.columns)

# Print the unique values in the 'gm_well_category' column of the selected DataFrame
print("Unique values in gm_well_category column:", df_nitrate_main.gm_well_category.unique())
# %%
#===================================================
# Preprocess to create input
#===================================================
df_nitrate = df_nitrate_main.copy()

# User input
min_sample = 10 # minimum sample size considered for trend analysis

# read data
df_nitrate.rename(columns = {'gm_well_id':'well_id', 'gm_latitude':'APPROXIMATE LATITUDE', 'gm_longitude':'APPROXIMATE LONGITUDE', 'gm_chemical_vvl': 'CHEMICAL', 'gm_result': 'RESULT','gm_well_category':'well_type','gm_samp_collection_date':'DATE'}, inplace = True)
df_nitrate['DATE']= pd.to_datetime(df_nitrate['DATE'])

# Convert the date column to a datetime object
df_nitrate["date"] = pd.to_datetime(df_nitrate["DATE"])

# Measurement start and end dates for different wells
start_date = df_nitrate.groupby("well_id")["date"].min()
end_date = df_nitrate.groupby("well_id")["date"].max()
dates_tog = pd.concat([start_date, end_date], axis=1)
dates_tog.columns = ['start_date','end_date']

# Create a new DataFrame with the statistics for each well
statistics = df_nitrate.groupby("well_id").agg({
    "RESULT": ["mean", "median", "max", "min"],
    "DATE": "count"
}).reset_index()

# Calculate the historic trend in data trend in data
#=====================================================
# Create a new DataFrame with the statistics for each well
trend_df_nitrate = pd.DataFrame(columns=['well_id', 'trend','change_per_year'])

for well_id, group in df_nitrate.groupby('well_id'):
    try:
        if len(group) < min_sample:
            trend_df_nitrate = trend_df_nitrate.append({'well_id': well_id, 'trend': f'sample_less_than_{min_sample}','change_per_year':0,'total_obs': len(group)}, ignore_index=True)
        else:
            group = group.sort_values(by='DATE')
            x = group['DATE'].apply(lambda x: x.toordinal())
            y = group['RESULT'].values
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            if p_value < 0.05:
                change_per_year = slope*365
                if(slope>0):
                    trend_df_nitrate = trend_df_nitrate.append({'well_id': well_id, 'trend': 'positive', 'change_per_year':change_per_year,'total_obs': len(group)}, ignore_index=True)
                elif(slope<0):
                    trend_df_nitrate = trend_df_nitrate.append({'well_id': well_id, 'trend': 'negative','change_per_year':change_per_year,'total_obs': len(group)}, ignore_index=True)
            else:
                trend_df_nitrate = trend_df_nitrate.append({'well_id': well_id, 'trend': 'not_significant','change_per_year':0,'total_obs': len(group)}, ignore_index=True)
    except ValueError:
        trend_df_nitrate = trend_df_nitrate.append({'well_id': well_id, 'trend': 'not_significant','change_per_year': 0, 'total_obs': len(group)}, ignore_index=True)

# positive_df_nitrate = trend_df_nitrate.query('trend == "positive"')
# negative_df_nitrate = trend_df_nitrate.query('trend == "negative"')

# Rename the columns in the new DataFrame
statistics.columns = ["well_id", "mean_nitrate", "median_nitrate", "max_nitrate", "min_nitrate", "measurement_count"]

# Add columns for the mean concentration for each period
df_nitrate["period"] = df_nitrate["DATE"].dt.year

periods = {
    "2015-2022": (2015, 2022),
    "2010-2015": (2010, 2015),
    "2005-2010": (2005, 2010),
    "2000-2005": (2000, 2005),
    "2000-2010": (2000, 2005),
    "2000-2022": (2000, 2022),
    "2010-2014": (2010, 2014),
    "2010-2022": (2010, 2022),
    "2007-2009": (2007, 2009),
    "2012-2015": (2012, 2015),
    "2019-2021": (2019, 2021),
    "2017-2018": (2017, 2018)

}

for period, (start, end) in periods.items():
    mask = (df_nitrate["period"] >= start) & (df_nitrate["period"] <= end)
    period_data = df_nitrate.loc[mask]
    if period_data.size == 0:
        df_nitrate.loc[mask, f"mean_concentration_{period}"] = float('nan')
    else:
        # grouped_data = period_data.groupby("well_id")
        grouped_data = period_data.groupby("well_id")
        mean_concentration = grouped_data["RESULT"].mean()#.reset_index()
        mean_concentration = pd.DataFrame(mean_concentration)
        mean_concentration.index.rename('ind', inplace = True)
        mean_concentration['well_id'] = mean_concentration.index
        mean_concentration.columns = [f"mean_concentration_{period}","well_id"]
        df_nitrate = pd.merge(df_nitrate, mean_concentration, on=["well_id"], how='left')
        df_nitrate.reset_index(drop=True, inplace=True)
        mask = (df_nitrate["period"] >= start) & (df_nitrate["period"] <= end)
        df_nitrate.loc[mask, f"mean_concentration_{period}"] = df_nitrate.loc[mask, f"mean_concentration_{period}"].fillna(float('nan'))

        count_data = grouped_data["well_id"].count()
        count_data.index.rename('ind', inplace = True)
        count_data = pd.DataFrame(count_data)
        count_data['tmp'] = count_data.index
        count_data.columns = [f"count_{period}","well_id"]
        df_nitrate = pd.merge(df_nitrate, count_data, on=["well_id"], how='left')
        df_nitrate.reset_index(drop=True, inplace=True)
        mask = (df_nitrate["period"] >= start) & (df_nitrate["period"] <= end)
        df_nitrate.loc[mask, f"mean_concentration_{period}"] = df_nitrate.loc[mask, f"mean_concentration_{period}"].where(df_nitrate[f"count_{period}"]>=5, float('nan'))


# Merge the statistics DataFrame with the original DataFrame
result = pd.merge(df_nitrate, statistics, on="well_id") #.merge(trend_df_nitrate, on="well_id")
well_type_tmp = result[['well_id','well_type']]

# Extract the columns we want to keep
result = result[["well_id", "APPROXIMATE LATITUDE", "APPROXIMATE LONGITUDE", "mean_nitrate", "median_nitrate", "max_nitrate", "min_nitrate", "measurement_count" ,*[f"mean_concentration_{period}" for period in periods.keys()]]]

# Group the data by well_id
grouped = result.groupby("well_id").mean()
grouped = pd.merge(grouped, dates_tog, on=["well_id"], how='left')
grouped = grouped.reset_index()

# Rounding to three decimals
grouped = grouped.round(3)

# Merging trend data
grouped = pd.merge(grouped, trend_df_nitrate, on="well_id")

well_type_tmp = well_type_tmp.drop_duplicates(subset='well_id', keep='first')
grouped = pd.merge(grouped, well_type_tmp, on="well_id")

# Export the result DataFrame to a CSV file
# result.to_csv(config.data_processed / f"gama_wellids_largestobs_totst_{str(largest_wells_no)}.csv")
grouped.to_csv(config.data_processed / "well_stats/gamanitrate_latest_stats.csv")

# %%

