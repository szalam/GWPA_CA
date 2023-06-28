#%%
#---------------------------
# Import necessary packages
#---------------------------
import sys
import ast
import ast
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, 'src')
import config


#%%
#---------------------------
# User input here
#---------------------------
# Resistance variables for different radii
res_var = [
    'Resistivity_lyrs_9_rad_2_miles'
]
# List of radii values
radii = ['2']

# Constants
rad_well = 2
gama_old_new = 2 # 1: earlier version, 2: latest version of GAMA
all_dom_flag = 1 # 1: All, 2: Domestic
if all_dom_flag == 2:
    well_type_select = {1: 'Domestic', 2: 'DOMESTIC'}.get(gama_old_new) 
else:
    well_type_select = 'All'

# %%
#---------------------------
# Functions 
#---------------------------
def sort_best_models(df, method):
    # Evaluating based on absolute difference in MAE and then R2 
    # between training and cross-validation
    if method == 'absolute_difference':
        df_sorted = df.sort_values(by=['abs_diff_mae_train_cv', 'abs_diff_r2_train_cv', 'test_set_mae', 'test_set_r2'], ascending=[True, True, True, False])

    # Evaluating based on absolute difference in MAE and then R2
    # between cross validation and test [Preferring this one]
    elif method == 'val_test_difference':
        # df_sorted = df.sort_values(by=['mean_test_score', 'abs_diff_mae_val_test', 'abs_diff_r2_val_test'], ascending=[False, True, True])
        df_sorted = df.sort_values(by=['mean_test_score'], ascending=[False])
    else:
        raise ValueError("Invalid sorting method. Please choose either 'absolute_difference' or 'val_test_difference'.")

    return df_sorted

def get_best_rf_model(filepath):
    # Load data
    df_main = pd.read_csv(filepath)

    # Prepare data
    df = df_main.copy()

    # Sort best models
    sorted_df = sort_best_models(df, method='val_test_difference')

    # Get top model
    top_model = sorted_df.iloc[0]

    # Fit the model using the best parameters
    best_params_str = top_model['params']
    best_params = ast.literal_eval(best_params_str)
    best_rf = RandomForestRegressor(**best_params)

    return top_model, best_rf

# Read dataset
def load_data(version):
    """Load data based on version"""
    filename = "Dataset_processed_GAMAlatest.csv" if version == 2 else "Dataset_processed.csv"
    df = pd.read_csv(config.data_processed / filename)
    return df

def filter_data(df, well_type,all_dom_flag):
    """Filter"""
    exclude_subregions = [14, 15, 10, 19, 18, 9, 6]
    if all_dom_flag == 2:
        df = df[df.well_type ==  well_type] 
    df = df[(df[f'thickness_abovCond_{round(.1*100)}_lyrs_9_rad_2miles'] <= 31) | (~df['SubRegion'].isin(exclude_subregions))]
    return df

def get_data_train_test(df, res_var):
    
    columns_to_keep = ['mean_nitrate','N_total',#'Average_ag_area'
                            'ProbDOpt5ppm_Shallow','ProbDOpt5ppm_Deep','ProbMn50ppb_Shallow','ProbMn50ppb_Deep','PrecipMinusETin_1971_2000_GWRP', #'area_wt_sagbi',
                            'CAML1990_natural_water','DTW60YrJurgens', 'HiWatTabDepMin', 'LateralPosition',  'RechargeAnnualmmWolock', 'RiverDist_NEAR','well_type_encoded',
                            f'{res_var}']
    
    df2 = df[columns_to_keep]
    df2 = df2.dropna()
    # 
    # Features
    X = df2.drop("mean_nitrate", axis=1)

    # Target
    y = df2["mean_nitrate"]

    y[y<=0]=.00001
    y = np.log(y)

    # Split the data into a training set and a testing set

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def train_predict(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    return y_pred_train, y_pred_test

# %%
#---------------------------
# Main
#---------------------------

# Create a list to hold the models
best_rf_models = []
best_rf_models_reduced = []

# Create a DataFrame to hold results and a dictionary to hold the models
results_df = pd.DataFrame(columns=['Radius', 'Training MAE', 'Training MAE [Reduced]','CV MAE','CV MAE [Reduced]', 'Test MAE',
    'Test MAE [Reduced]', 'Training R2','Training R2 [Reduced]', 'CV R2','CV R2 [Reduced]', 'Test R2','Test R2 [Reduced]'
])
results_df_reduced = results_df.copy()

for res, radius in zip(res_var, radii):
    # File paths
    filepath = f'/Users/szalam/Main/00_Research_projects/GWPA_rf_outputs/rfmodel_config/rfmodel_parameters_resistivityUse_{res}.csv'
    filepath_reduced = f'/Users/szalam/Main/00_Research_projects/GWPA_rf_outputs/rfmodel_config/rfmodel_parameters_resistivityUse_{res}_reduced.csv'

    # Get the models
    top_model, best_rf = get_best_rf_model(filepath)
    top_model_reduced, best_rf_reduced = get_best_rf_model(filepath_reduced)

    # Add the model to the list
    best_rf_models.append(best_rf)
    best_rf_models_reduced.append(best_rf_reduced)
    
# Display results
# print(results_df)

# %%
# Load and process data
df_main = load_data(gama_old_new)
df = df_main[df_main.well_data_source == 'GAMA'].copy()

df['well_type_encoded'] = pd.factorize(df['well_type'])[0]
df['well_type_encoded'] = df['well_type_encoded'].where(df['well_type'].notna(), df['well_type'])

# separate wells inside cv
well_cv = pd.read_csv(config.data_processed / 'wells_inside_CV_GAMAlatest.csv',index_col=False)

# Assuming df is a dataframe with all wells
df_cv = df[df['well_id'].isin(well_cv['well_id'])]

df = filter_data(df_cv, well_type_select,all_dom_flag)

df = df.drop(['well_id', 'well_data_source','start_date', 'end_date'], axis=1)

#%%
for i, (res, radius) in enumerate(zip(res_var, radii)):

    print(i)

    X_train, X_test, y_train, y_test = get_data_train_test(df, res_var = res)

    best_rf = best_rf_models[i]
    best_rf.fit(X_train, y_train)

    # Train and predict
    y_pred_train_rf, y_pred_test_rf = train_predict(best_rf, X_train, y_train, X_test)

    # Calculate the metrics on train and test data
    mae_train_rf, rmse_train_rf, r2_train_rf = calculate_metrics(y_train, y_pred_train_rf)
    mae_test_rf, rmse_test_rf, r2_test_rf = calculate_metrics(y_test, y_pred_test_rf)

    # Calculate cross-validation scores
    cv_r2_rf = cross_val_score(best_rf, X_train, y_train, cv=3, scoring='r2').mean()
    cv_mae_rf = -cross_val_score(best_rf, X_train, y_train, cv=3, scoring='neg_mean_absolute_error').mean() 

    #======================================================
    # Reduced version
    #======================================================
    # Remove the selected features from the training and testing sets
    X_train_reduced = X_train.drop([f'{res}'], axis=1)
    X_test_reduced = X_test.drop([f'{res}'], axis=1)

    # Train and predict on the reduced data
    y_pred_train_rf_reduced, y_pred_test_rf_reduced = train_predict(best_rf_reduced, X_train_reduced, y_train, X_test_reduced)

    # Calculate the metrics on reduced train and test data
    mae_train_rf_reduced, rmse_train_rf_reduced, r2_train_rf_reduced = calculate_metrics(y_train, y_pred_train_rf_reduced)
    mae_test_rf_reduced, rmse_test_rf_reduced, r2_test_rf_reduced = calculate_metrics(y_test, y_pred_test_rf_reduced)

    # Calculate cross-validation scores for reduced dataset
    cv_r2_rf_reduced = cross_val_score(best_rf_reduced, X_train_reduced, y_train, cv=3, scoring='r2').mean()
    cv_mae_rf_reduced = -cross_val_score(best_rf_reduced, X_train_reduced, y_train, cv=3, scoring='neg_mean_absolute_error').mean() 

    new_row = pd.DataFrame({
        'Radius': [radius],
        'Training MAE': [mae_train_rf],
        'Training MAE [Reduced]': [mae_train_rf_reduced],
        'CV MAE': [cv_mae_rf],
        'CV MAE [Reduced]': [cv_mae_rf_reduced],
        'Test MAE': [mae_test_rf],
        'Test MAE [Reduced]': [mae_test_rf_reduced],
        'Training R2': [r2_train_rf],
        'Training R2 [Reduced]': [r2_train_rf_reduced],
        'CV R2': [cv_r2_rf],
        'CV R2 [Reduced]': [cv_r2_rf_reduced],
        'Test R2': [r2_test_rf],
        'Test R2 [Reduced]': [r2_test_rf_reduced]
    })

    results_df_reduced = pd.concat([results_df_reduced, new_row], ignore_index=True)

print(results_df_reduced)
# %%
filepath = Path('/Users/szalam/Main/00_Research_projects/GWPA_rf_outputs/rfmodel_config')

# Round to three decimal places, Transpose the DataFrame
results_df_reduced = results_df_reduced.round(3)
results_df_reduced_transposed = results_df_reduced.T

# Save the transposed DataFrame to a CSV file
results_df_reduced_transposed.to_csv(filepath / 'rfmetrics_full_reduced.csv', header=False)



#%%
#=============================================================================
# Estimate the MAE and R2 for all random forest parameters
# The goal is to estimate the difference in MAE between CV and test set
#=============================================================================
filepath = f'/Users/szalam/Main/00_Research_projects/GWPA_rf_outputs/rfmodel_config/rfmodel_parameters_resistivityUse_Resistivity_lyrs_9_rad_2_miles.csv'
cv_results = pd.read_csv(filepath)

# Specify the start and end points for the loop
start_point = 455
end_point = len(cv_results['params'])

# %%
filepath = Path('/Users/szalam/Main/00_Research_projects/GWPA_rf_outputs/rfmodel_config')

# Loop from the start point to the end point
for i in range(start_point, end_point):
    print(i)
    params = cv_results['params'][i]
    params_dict = ast.literal_eval(params)
    model = RandomForestRegressor(**params_dict, random_state=42, bootstrap=True)
    model.fit(X_train, y_train)

    # Training dataset metrics
    y_pred_train = model.predict(X_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    cv_results.at[i, 'train_set_mae'] = mae_train
    cv_results.at[i, 'train_set_r2'] = r2_train

    # Test dataset metrics
    y_pred_test = model.predict(X_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    
    cv_results.at[i, 'test_set_mae'] = mae_test
    cv_results.at[i, 'test_set_r2'] = r2_test
    cv_results.at[i, 'test_set_score'] = mse_test

    # CV dataset metrics
    # Compute the mean MAE via cross-validation
    neg_mae_cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_absolute_error')
    cv_results.at[i, 'cv_mae'] = -np.mean(neg_mae_cv_scores)
    
    # Compute the mean R2 via cross-validation
    r2_cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
    cv_results.at[i, 'cv_r2'] = np.mean(r2_cv_scores)
# %%
# Save the transposed DataFrame to a CSV file
# cv_results.to_csv(filepath + 'rfmetrics_full_wt_r2_mae_all_rfparams2.csv', header=True)

# %%
# cv_results_2 = cv_results.copy()
# cv_results_2 = pd.read_csv(filepath + 'rfmetrics_full_wt_r2_mae_all_rfparams')
# Read the first CSV file
df_tmp1 = pd.read_csv(filepath / 'rfmetrics_full_wt_r2_mae_all_rfparams.csv')
df_tmp2 = pd.read_csv(filepath / 'rfmetrics_full_wt_r2_mae_all_rfparams2.csv')

# Select rows from df_tmp1 and df_tmp2
df1_selected = df_tmp1.iloc[:455]
df2_selected = df_tmp2.iloc[455:]

# Combine the two dataframes based on matching column names
cv_results_2 = pd.concat([df1_selected, df2_selected], axis=0, ignore_index=True)

# %%
cv_results_2.columns
# %%
# Calculate the absolute differences between 'test_set_mae' and 'cv_mae'
cv_results_2['mae_difference'] = abs(cv_results_2['test_set_mae'] - cv_results_2['cv_mae'])

# Print the range of the differences
print("Range of absolute differences in MAE:", cv_results_2['mae_difference'].min(), "to", cv_results_2['mae_difference'].max())

# Calculate the difference in percentage compared to 'cv_mae'
cv_results_2['mae_difference_percentage'] = abs(cv_results_2['mae_difference'] / cv_results_2['cv_mae'] * 100)

# Print the mean of the percentage differences
print("Mean of the percentage differences in MAE:", cv_results_2['mae_difference_percentage'].min(), "%", "to", cv_results_2['mae_difference_percentage'].max(), "%")

# Get the row with rank_test_score = 1
best_row = cv_results_2[cv_results_2['rank_test_score'] == 1]

# Print the absolute difference and mae_difference_percentage for this row
print("Absolute difference in MAE for the best model:", best_row['mae_difference'].values[0])
print("Percentage difference in MAE for the best model:", best_row['mae_difference_percentage'].values[0], "%")

# %%
