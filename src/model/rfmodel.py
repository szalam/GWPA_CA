#%%
import sys
sys.path.insert(0,'src')

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import config
import numpy as np
#%%

# Read dataset
df = pd.read_csv(config.data_processed / "Dataset_processed.csv")
df = df.drop(['well_id', 'well_data_source','start_date', 'end_date'], axis=1)


#%%
df['nitrate_increase'] = df['mean_concentration_2015-2022']- df['mean_concentration_2005-2010']

#%%
# df2 = df.drop(columns=['median_nitrate', 'max_nitrate', 
#        'min_nitrate', 'measurement_count','mean_concentration_2015-2022', 
#        'mean_concentration_2010-2015','mean_concentration_2005-2010', 
#        'mean_concentration_2000-2005', 'mean_concentration_2000-2022', 
#        'mean_concentration_2010-2022', 'mean_concentration_2007-2009', 
#        'mean_concentration_2012-2015', 'mean_concentration_2019-2021', 
#        'mean_concentration_2017-2018',
#        'APPROXIMATE LATITUDE', 'APPROXIMATE LONGITUDE','nitrate_increase','Cafo_Population'])

df2 = df[['mean_nitrate','Conductivity','area_wt_sagbi','gwdep','Average_ag_area','Cafo_Population_5miles']]
#%%
# # select rows where the values in the 'total_ag_2010' column are greater than zero
# df2 = df2.query('total_ag_2010 > 0')

#%%# Remove rows with missing values
df2 = df2.dropna()

# Select features and target for the model
X = df2.drop(columns=['mean_nitrate'])
y = df2['mean_nitrate']

#%%
# Count the number of missing values in each column
missing_values = X.isna().sum()

# Print the number of missing values in each column
print(missing_values)

# Find the column with the highest number of missing values
highest_missing = missing_values[missing_values == np.max(missing_values)].index[0]

# Print the column name with the highest number of missing values
if highest_missing == 0:
    print("No missing value")
else:
    print("Column with highest missing values: ", highest_missing)

#%%
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the random forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf.predict(X_test)

# Calculate the mean absolute error of the predictions
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error: ", mae)

# %%
# Get feature importances
importances = rf.feature_importances_

# Get feature names
feature_names = X.columns

# Create a dataframe of feature importances
feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})

# Sort the dataframe by feature importances
feature_importances.sort_values(by='importance', ascending=False, inplace=True)

# Plot the feature importances
plt.barh(feature_importances['feature'], feature_importances['importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()

# %%
feature_importances = feature_importances.sort_values(by='importance', ascending=False)
# %%
