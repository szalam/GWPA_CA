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
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
#%%
rad_well = 2 # mile
#%%
# Read dataset
df = pd.read_csv(config.data_processed / "Dataset_processed.csv")
df = df[df.well_data_source == 'GAMA']
df = df.drop(['well_id', 'well_data_source','start_date', 'end_date'], axis=1)

# Assuming df is your DataFrame with NaN values in the 'Cafo_Population_5miles' column
df['CAFO_Population_5miles'] = df['CAFO_Population_5miles'].fillna(0)
df = df[df.measurement_count>10]
# df = df[df.well_type == 'Domestic']
# Remove high salinity regions
# exclude_subregions = [14, 15, 10, 19,18, 9,6]
# # filter aemres to keep only the rows where Resistivity is >= 10 and SubRegion is not in the exclude_subregions list
# df = df[(df[f'thickness_abovCond_{round(.1*100)}_lyrs_9_rad_{rad_well}miles'] <= 31) | (~df['SubRegion'].isin(exclude_subregions))]

#%%
# df['nitrate_increase'] = df['mean_concentration_2015-2022']- df['mean_concentration_2005-2010']

def rf_model(df, flag_option, rad_well):
    
    if flag_option == 1:
        columns_to_keep = ['mean_nitrate','Average_ag_area','N_total',#'area_wt_sagbi',
                        'ProbDOpt5ppm_Shallow','ProbDOpt5ppm_Deep','ProbMn50ppb_Shallow','ProbMn50ppb_Deep',#'CAFO_Population_5miles', 'Ngw_1975'
                        'CAML1990_natural_water','DTW60YrJurgens', 'HiWatTabDepMin', 'LateralPosition', 'RechargeAnnualmmWolock', 'RiverDist_NEAR'] # ,'Cafo_Population_5miles'
        df2 = df[columns_to_keep]

        # Removing 0 or negative values for log conversion
        df2 = df2.applymap(lambda x: np.nan if x <= 0 else x)
        df2.mean_nitrate = np.log(df2['mean_nitrate'])

    if flag_option == 2:
        columns_to_keep = ['mean_nitrate','Average_ag_area','N_total',#'area_wt_sagbi'
                        'ProbDOpt5ppm_Shallow','ProbDOpt5ppm_Deep','ProbMn50ppb_Shallow','ProbMn50ppb_Deep',#'CAFO_Population_5miles','Ngw_1975',
                        'CAML1990_natural_water','DTW60YrJurgens', 'HiWatTabDepMin', 'LateralPosition',  'RechargeAnnualmmWolock', 'RiverDist_NEAR', #'PrecipMinusETin_1971_2000_GWRP',
                        'Resistivity_lyrs_9','Resistivity_lyrs_6','Resistivity_lyrs_4'
                        ] # ,'Cafo_Population_5miles', 'gwdep',

        for i in range(2,21):
            column_name = f'Conductivity_depthwtd_lyr{i}_rad_2mile'
            # column_name2 = f'Conductivity_depthwtd_lyr{i}_rad_1mile'
            columns_to_keep.append(column_name)
            # columns_to_keep.append(column_name2)
        df2 = df[columns_to_keep]
        na_counts = df2[columns_to_keep].isna().sum()
        print(na_counts)

        # Create a dictionary of old and new column names
        rename_dict = {}
        for column in columns_to_keep:
            new_column_name = column.replace('Conductivity_depthwtd_lyr', '(Conductivity x thickness) for lyr')
            rename_dict[column] = new_column_name

        # Rename the columns
        df2 = df[columns_to_keep].rename(columns=rename_dict)

        # Removing 0 or negative values for log conversion
        df2 = df2.applymap(lambda x: np.nan if x <= 0 else x)
        df2.mean_nitrate = np.log(df2['mean_nitrate'])

    if flag_option == 3:
        columns_to_keep = ['mean_nitrate','N_total',
                        'ProbMn50ppb_Shallow','ProbMn50ppb_Deep','DTW60YrJurgens', 'RechargeAnnualmmWolock',
                        'Resistivity_lyrs_9'] #'PrecipMinusETin_1971_2000_GWRP' ,'Cafo_Population_5miles'
        df2 = df[columns_to_keep]

        # Removing 0 or negative values for log conversion
        df2 = df2.applymap(lambda x: np.nan if x <= 0 else x)
        df2.mean_nitrate = np.log(df2['mean_nitrate'])

    if flag_option == 4:
        columns_to_keep = ['mean_nitrate',
                        'Resistivity_lyrs_9','Resistivity_lyrs_6','Resistivity_lyrs_4'] #'PrecipMinusETin_1971_2000_GWRP' ,'Cafo_Population_5miles'
        df2 = df[columns_to_keep]

        for i in range(2,21):
            column_name = f'Conductivity_depthwtd_lyr{i}_rad_2mile'
            # column_name2 = f'Conductivity_depthwtd_lyr{i}_rad_1mile'
            columns_to_keep.append(column_name)
            # columns_to_keep.append(column_name2)

        df2 = df[columns_to_keep]
        na_counts = df2[columns_to_keep].isna().sum()
        print(na_counts)

        # Create a dictionary of old and new column names
        rename_dict = {}
        for column in columns_to_keep:
            new_column_name = column.replace('Conductivity_depthwtd_lyr', '(Conductivity x thickness) for lyr')
            rename_dict[column] = new_column_name

        # Rename the columns
        df2 = df[columns_to_keep].rename(columns=rename_dict)

        # Removing 0 or negative values for log conversion
        df2 = df2.applymap(lambda x: np.nan if x <= 0 else x)
        df2.mean_nitrate = np.log(df2['mean_nitrate'])

    if flag_option == 5:
        columns_to_keep = ['mean_nitrate','Average_ag_area','N_total',#'area_wt_sagbi'
                        'ProbDOpt5ppm_Shallow','ProbDOpt5ppm_Deep','ProbMn50ppb_Shallow','ProbMn50ppb_Deep',#'CAFO_Population_5miles','Ngw_1975',
                        'CAML1990_natural_water','DTW60YrJurgens', 'HiWatTabDepMin', 'LateralPosition',  'RechargeAnnualmmWolock', 'RiverDist_NEAR', #'PrecipMinusETin_1971_2000_GWRP',
                        'Conductivity_depthwtd_lyr2_rad_2mile','Conductivity_depthwtd_lyr20_rad_2mile'
                        ] 
        df2 = df[columns_to_keep]

        # Create a dictionary of old and new column names
        rename_dict = {}
        for column in columns_to_keep:
            new_column_name = column.replace('Conductivity_depthwtd_lyr', '(Conductivity x thickness) for lyr')
            rename_dict[column] = new_column_name

        # Rename the columns
        df2 = df[columns_to_keep].rename(columns=rename_dict)

        # Removing 0 or negative values for log conversion
        df2 = df2.applymap(lambda x: np.nan if x <= 0 else x)
        df2.mean_nitrate = np.log(df2['mean_nitrate'])
    
    # # Remove rows with missing values
    df2 = df2.dropna()

    new_column_names = {'area_wt_sagbi': 'SAGBI weighted by area',
                    'Average_ag_area': 'Avg Agricultural area',
                    'N_total': 'Nitrogen input from landscape',
                    'ProbDOpt5ppm_Shallow': 'Prob. DO below 5ppm [Shallow]',
                    'ProbDOpt5ppm_Deep': 'Prob. DO below 5ppm [Deep]',
                    'ProbMn50ppb_Shallow': 'Prob. of Mn above 50ppb [Shallow]',
                    'ProbMn50ppb_Deep': 'Prob. of Mn above 50ppb [Deep]',
                    'CAML1990_natural_water': 'Percent landuse as natural or water (1990)',
                    'DTW60YrJurgens': 'Depth to 60yr old groundwater',
                    'HiWatTabDepMin': 'Depth to seasonal high water table [lowest]',
                    'LateralPosition': 'Normalized distance from valley axis',
                    'Ngw_1975': 'Normalized unsaturated zone nitrogen load to water table',
                    'RechargeAnnualmmWolock': 'Mean annual natural groundwater recharge (mm/year)',
                    'RiverDist_NEAR': 'Distance to river with stream order 3 ',
                    'PrecipMinusETin_1971_2000_GWRP': 'Precipitation - ET during 1971-2000',
                    'Resistivity_lyrs_9': 'Resistivity 9 layer',
                    'Resistivity_lyrs_6': 'Resistivity 6 layer',
                    'Resistivity_lyrs_4': 'Resistivity 4 layer'}

    df2 = df2.rename(columns=new_column_names)

    # Select features and target for the model
    X = df2.drop(columns=['mean_nitrate'])
    y = df2['mean_nitrate']

    
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

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit the random forest model
    rf = RandomForestRegressor(n_estimators=150, random_state=42)
    rf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf.predict(X_test)

    return y_test, y_pred, X, y, rf

y1_test, y1_pred, X1, y1, rf1 = rf_model(df, flag_option=1, rad_well=2)
y2_test, y2_pred, X2, y2, rf2 = rf_model(df, flag_option=2, rad_well=2)
y3_test, y3_pred, X3, y3, rf3 = rf_model(df, flag_option=3, rad_well=2)
y4_test, y4_pred, X4, y4, rf4 = rf_model(df, flag_option=4, rad_well=2)
y5_test, y5_pred, X5, y5, rf5 = rf_model(df, flag_option=5, rad_well=2)

#%%
# Calculate the mean absolute error of the predictions
mae1 = mean_absolute_error(y1_test, y1_pred)
mae2 = mean_absolute_error(y2_test, y2_pred)
mae3 = mean_absolute_error(y3_test, y3_pred)
mae4 = mean_absolute_error(y4_test, y4_pred)
mae5 = mean_absolute_error(y5_test, y5_pred)
print("Mean Absolute Error: ", mae1)
print("Mean Absolute Error: ", mae2)
print("Mean Absolute Error: ", mae3)
print("Mean Absolute Error: ", mae4)
print("Mean Absolute Error: ", mae5)

mse_rf1 = mean_squared_error(y1_test, y1_pred)
mse_rf2 = mean_squared_error(y2_test, y2_pred)
mse_rf3 = mean_squared_error(y3_test, y3_pred)
mse_rf4 = mean_squared_error(y4_test, y4_pred)
mse_rf5 = mean_squared_error(y5_test, y5_pred)
print(f"MSE for RF1: {mse_rf1:.4f}")
print(f"MSE for RF2: {mse_rf2:.4f}")
print(f"MSE for RF3: {mse_rf3:.4f}")
print(f"MSE for RF4: {mse_rf4:.4f}")
print(f"MSE for RF5: {mse_rf5:.4f}")

r2_rf1 = r2_score(y1_test, y1_pred)
r2_rf2 = r2_score(y2_test, y2_pred)
r2_rf3 = r2_score(y3_test, y3_pred)
r2_rf4 = r2_score(y4_test, y4_pred)
r2_rf5 = r2_score(y5_test, y5_pred)
print(f"R-squared for RF1: {r2_rf1:.4f}")
print(f"R-squared for RF2: {r2_rf2:.4f}")
print(f"R-squared for RF2: {r2_rf3:.4f}")
print(f"R-squared for RF2: {r2_rf4:.4f}")
print(f"R-squared for RF2: {r2_rf5:.4f}")

#%%
def feature_imp_plot(rf, X):
    # Get feature importances
    importances = rf.feature_importances_

    # Get feature names
    feature_names = X.columns

    # Create a dataframe of feature importances
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})

    # Sort the dataframe by feature importances
    feature_importances.sort_values(by='importance', ascending=False, inplace=True)

    # Plot the feature importances
    # Set the figure size
    plt.figure(figsize=(10, 12))
    plt.barh(feature_importances['feature'], feature_importances['importance'])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.show()

    return feature_importances
#%%
feature_importances1 = feature_imp_plot(rf = rf1, X = X1)
feature_importances2 = feature_imp_plot(rf = rf2, X = X2)
feature_importances3 = feature_imp_plot(rf = rf3, X = X3)
feature_importances4 = feature_imp_plot(rf = rf4, X = X4)
feature_importances5 = feature_imp_plot(rf = rf5, X = X5)

# %%
feature_importances1 = feature_importances1.sort_values(by='importance', ascending=False)
# %%
def plot_obs_pred_rf(y_test,y_pred):
    # Set seaborn aesthetic parameters
    sns.set(style='whitegrid', palette='pastel')

    # Create a scatterplot of observed mean_nitrate vs predicted mean_nitrate
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)

    # Set the x-axis label
    plt.xlabel('Observed log(Mean Nitrate)', fontsize=14)

    # Set the y-axis label
    plt.ylabel('Predicted log(Mean Nitrate)', fontsize=14)

    # Set the title
    plt.title('Observed vs Predicted Mean Nitrate', fontsize=16)

    # Add a diagonal line to visualize the perfect match
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)

    # Remove the top and right spines
    sns.despine(top=True, right=True, left=False, bottom=False)

    # Show the plot
    plt.show()
# %%
plot_obs_pred_rf(y1_test,y1_pred)
plot_obs_pred_rf(y2_test,y2_pred)
plot_obs_pred_rf(y3_test,y3_pred)
plot_obs_pred_rf(y4_test,y4_pred)
plot_obs_pred_rf(y5_test,y5_pred)

# %%
# Marginal predictive power of resistivity related variables

def marginal_predictive(feature_importances,X):

    feature_names = X.columns.tolist()
    # extract the indices of input features that have "Resistivity" or "Conductivity" in their name
    resistivity_conductivity_indices = [i for i in range(len(feature_importances)) if "Resistivity" in feature_names[i] or "Conductivity" in feature_names[i]]

    # calculate the marginal predictive power of the 3 resistivity-related variables
    marginal_predictive_power = np.sum(feature_importances.iloc[resistivity_conductivity_indices].importance)

    print("Marginal predictive power of resistivity-related variables:", marginal_predictive_power)
# %%
marginal_predictive(feature_importances2,X2)
marginal_predictive(feature_importances1,X2)
marginal_predictive(feature_importances4,X4)
marginal_predictive(feature_importances5,X5)
# %%
