#===================================================================================================
# This script reads a CSV file with GAMA groundwater well data time series,
# then it calculates the following statistics for each well:
# (1) mean, median, max, min nitrate concentration, 
# (2) number of nitrate measurement, 
# (3) mean concentration for the period 2015-2022, 2010-2015, 2000-2010. Avoids mean calculation when obs no<5 for a period
# The script exports the resulting dataframe to a new CSV file. 
# Exported data also includes well id, lat/lon, and the calculated statistics in different columns.
#===================================================================================================
#%%
import sys
sys.path.insert(0,'src')
import pandas as pd
import config
import ppfun as dp

# file location
file_polut = config.data_raw / "nitrate_data/UCDNitrateData.csv"

# read data
# read gama excel file
df = pd.read_excel(config.data_gama / 'TULARE_NO3N.xlsx',engine='openpyxl')
df.rename(columns = {'GM_WELL_ID':'well_id', 'GM_LATITUDE':'APPROXIMATE LATITUDE', 'GM_LONGITUDE':'APPROXIMATE LONGITUDE', 'GM_CHEMICAL_VVL': 'CHEMICAL', 'GM_RESULT': 'RESULT','GM_WELL_CATEGORY':'DATASET_CAT','GM_SAMP_COLLECTION_DATE':'DATE'}, inplace = True)
df['DATE']= pd.to_datetime(df['DATE'])


# Convert the date column to a datetime object
df["date"] = pd.to_datetime(df["DATE"])

# Measurement start and end dates for different wells
start_date = df.groupby("well_id")["date"].min()
end_date = df.groupby("well_id")["date"].max()
dates_tog = pd.concat([start_date, end_date], axis=1)
dates_tog.columns = ['start_date','end_date']

# Create a new DataFrame with the statistics for each well
statistics = df.groupby("well_id").agg({
    "RESULT": ["mean", "median", "max", "min"],
    "DATE": "count"
}).reset_index()

# Rename the columns in the new DataFrame
statistics.columns = ["well_id", "mean_nitrate", "median_nitrate", "max_nitrate", "min_nitrate", "measurement_count"]

# Add columns for the mean concentration for each period
df["period"] = df["DATE"].dt.year

periods = {
    "2015-2022": (2015, 2022),
    "2010-2015": (2010, 2015),
    "2005-2010": (2005, 2010),
    "2000-2005": (2000, 2005),
    "2000-2022": (2000, 2022),
    "2010-2022": (2010, 2022),
    "2007-2009": (2007, 2009),
    "2012-2015": (2012, 2015),
    "2019-2021": (2019, 2021),
    "2017-2018": (2017, 2018)

}

for period, (start, end) in periods.items():
    mask = (df["period"] >= start) & (df["period"] <= end)
    period_data = df.loc[mask]
    if period_data.size == 0:
        df.loc[mask, f"mean_concentration_{period}"] = float('nan')
    else:
        # grouped_data = period_data.groupby("well_id")
        grouped_data = period_data.groupby("well_id")
        mean_concentration = grouped_data["RESULT"].mean()#.reset_index()
        mean_concentration = pd.DataFrame(mean_concentration)
        mean_concentration.index.rename('ind', inplace = True)
        mean_concentration['well_id'] = mean_concentration.index
        mean_concentration.columns = [f"mean_concentration_{period}","well_id"]
        df = pd.merge(df, mean_concentration, on=["well_id"], how='left')
        df.loc[mask, f"mean_concentration_{period}"] = df.loc[mask, f"mean_concentration_{period}"].fillna(float('nan'))
        
        count_data = grouped_data["well_id"].count()
        count_data.index.rename('ind', inplace = True)
        count_data = pd.DataFrame(count_data)
        count_data['tmp'] = count_data.index
        count_data.columns = [f"count_{period}","well_id"]
        df = pd.merge(df, count_data, on=["well_id"], how='left')
        df.loc[mask, f"mean_concentration_{period}"] = df.loc[mask, f"mean_concentration_{period}"].where(df[f"count_{period}"]>=5, float('nan'))


# Merge the statistics DataFrame with the original DataFrame
result = pd.merge(df, statistics, on="well_id")

# Extract the columns we want to keep
result = result[["well_id", "APPROXIMATE LATITUDE", "APPROXIMATE LONGITUDE", "mean_nitrate", "median_nitrate", "max_nitrate", "min_nitrate", "measurement_count", *[f"mean_concentration_{period}" for period in periods.keys()]]]

# Group the data by well_id
grouped = result.groupby("well_id").mean()
grouped = pd.merge(grouped, dates_tog, on=["well_id"], how='left')
grouped = grouped.reset_index()

# Rounding to three decimals
grouped = grouped.round(3)

# Export the result DataFrame to a CSV file
# result.to_csv(config.data_processed / f"gama_wellids_largestobs_totst_{str(largest_wells_no)}.csv")
grouped.to_csv(config.data_processed / "well_stats/gamanitrate_stats.csv")

# %%
