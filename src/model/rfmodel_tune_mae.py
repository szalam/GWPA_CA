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
rad_well = 2
# Read dataset
df = pd.read_csv(config.data_processed / "Dataset_processed.csv")
df = df[df.well_data_source == 'GAMA']
df = df.drop(['well_id', 'well_data_source','start_date', 'end_date'], axis=1)

# df = df[df.measurement_count>5]
# df = df[df.well_type == 'Domestic']
# Remove high salinity regions
exclude_subregions = [14, 15, 10, 19,18, 9,6]
# # filter aemres to keep only the rows where Resistivity is >= 10 and SubRegion is not in the exclude_subregions list
df = df[(df[f'thickness_abovCond_{round(.1*100)}_lyrs_9_rad_{rad_well}miles'] <= 31) | (~df['SubRegion'].isin(exclude_subregions))]

# %%

def rf_mod_compare(df, res_var = 'Resistivity_lyrs_9_rad_0_2_miles'):
    import numpy as np
    columns_to_keep = ['mean_nitrate','Average_ag_area','N_total',#'area_wt_sagbi'
                            'ProbDOpt5ppm_Shallow','ProbDOpt5ppm_Deep','ProbMn50ppb_Shallow','ProbMn50ppb_Deep','PrecipMinusETin_1971_2000_GWRP', 'area_wt_sagbi',
                            'CAML1990_natural_water','DTW60YrJurgens', 'HiWatTabDepMin', 'LateralPosition',  'RechargeAnnualmmWolock', 'RiverDist_NEAR',
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

    # Step 4: Split the data into a training set and a testing set

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    X_train.shape, X_test.shape, y_train.shape, y_test.shape
    
    # Step 5: Train a Random Forest model

    from sklearn.ensemble import RandomForestRegressor

    # Create a Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42,bootstrap=True)

    # Train the model
    rf.fit(X_train, y_train)

    # 
    #==================================
    # Tuning via grid search
    #==================================
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import r2_score
    from time import time

    import numpy as np

    # Define the hyperparameters
    rf_params = {
        'n_estimators': [500,1000,2000],
        'max_depth': [15, 25],
        'min_samples_split': [5, 15, 25, 50],
        'min_samples_leaf': [10, 25, 50],
        'max_features' : ['sqrt', 0.25, 0.5],

    }
    
    # Create a GridSearchCV object
    grid_search_rf = GridSearchCV(estimator=rf, param_grid=rf_params, cv=3,
                                scoring='neg_mean_squared_error',
                                #   scoring='r2',   
                                  return_train_score=True, # set this to True to check for overfitting
                                  verbose=3,n_jobs=-1)

    start_time = time()
    # Perform the grid search
    grid_search_rf.fit(X_train, y_train)
    cv_results = pd.DataFrame.from_dict(grid_search_rf.cv_results_)
    cv_results.to_csv(config.data_processed / f'rfmodel_config/rfmodel_parameters_resistivityUse_{res_var}.csv', index=False)

    total_time = time() - start_time

    # Get the best model
    best_rf = grid_search_rf.best_estimator_

    # Evaluate the model
    y_pred_train_rf = best_rf.predict(X_train)
    y_pred_test_rf = best_rf.predict(X_test)

    # Calculate the metrics
    mae_train_rf = mean_absolute_error(np.exp(y_train), np.exp(y_pred_train_rf))
    rmse_train_rf = np.sqrt(mean_squared_error(np.exp(y_train), np.exp(y_pred_train_rf)))

    mae_test_rf = mean_absolute_error(np.exp(y_test), np.exp(y_pred_test_rf))
    rmse_test_rf = np.sqrt(mean_squared_error(np.exp(y_test), np.exp(y_pred_test_rf)))

    # best_rf, mae_train_rf, rmse_train_rf, mae_test_rf, rmse_test_rf

    # 
    # #==================================
    # # Tuning via randomized search
    # #==================================
    # from sklearn.model_selection import RandomizedSearchCV

    # # Define the hyperparameters
    # rf_params = {
    #     'n_estimators': [50, 100, 200],
    #     'max_depth': [None, 10, 20, 30],
    #     'min_samples_split': [2, 10, 25],
    #     'min_samples_leaf': [1, 10, 25]
    # }

    # # Create a RandomizedSearchCV object
    # random_search_rf = RandomizedSearchCV(estimator=rf, param_distributions=rf_params, 
    #                                     n_iter=20, cv=3, scoring='neg_mean_squared_error', 
    #                                     verbose=2, random_state=42, n_jobs=-1)

    # # Perform the random search
    # random_search_rf.fit(X_train, y_train)

    # # Get the best model
    # best_rf = random_search_rf.best_estimator_

    # #
    # # Evaluate the model
    # y_pred_train_rf = best_rf.predict(X_train)
    # y_pred_test_rf = best_rf.predict(X_test)

    # # Calculate the metrics
    # mae_train_rf = mean_absolute_error(np.exp(y_train), np.exp(y_pred_train_rf))
    # rmse_train_rf = np.sqrt(mean_squared_error(np.exp(y_train), np.exp(y_pred_train_rf)))

    # mae_test_rf = mean_absolute_error(np.exp(y_test), np.exp(y_pred_test_rf))
    # rmse_test_rf = np.sqrt(mean_squared_error(np.exp(y_test), np.exp(y_pred_test_rf)))

    # best_rf, mae_train_rf, rmse_train_rf, mae_test_rf, rmse_test_rf


    # 
    #===========================================
    # Tune without resistivity related variables
    #===========================================
    # Remove the selected features from the training and testing sets
    X_train_reduced = X_train.drop([f'{res_var}'], axis=1)
    X_test_reduced = X_test.drop([f'{res_var}'], axis=1)

    # Retrain the model on the reduced training data
    best_rf.fit(X_train_reduced, y_train)

    # Evaluate the model on the reduced training data
    y_pred_train_rf_reduced = best_rf.predict(X_train_reduced)
    mae_train_rf_reduced = mean_absolute_error(np.exp(y_train), np.exp(y_pred_train_rf_reduced))
    rmse_train_rf_reduced = np.sqrt(mean_squared_error(np.exp(y_train), np.exp(y_pred_train_rf_reduced)))

    # Evaluate the model on the reduced testing data
    y_pred_test_rf_reduced = best_rf.predict(X_test_reduced)
    mae_test_rf_reduced = mean_absolute_error(np.exp(y_test), np.exp(y_pred_test_rf_reduced))
    rmse_test_rf_reduced = np.sqrt(mean_squared_error(np.exp(y_test), np.exp(y_pred_test_rf_reduced)))

    # mae_train_rf_reduced, rmse_train_rf_reduced, mae_test_rf_reduced, rmse_test_rf_reduced


    # 
    diff_mae_train = mae_train_rf_reduced-mae_train_rf
    diff_rmse_train = rmse_train_rf_reduced-rmse_train_rf
    diff_mae_test = mae_test_rf_reduced-mae_test_rf
    diff_rmse_test = rmse_test_rf_reduced-rmse_test_rf
    # Difference with and without resistivity
    print('+ve means error increased when resistivity related vars removed')
    print(diff_mae_train)
    print(diff_rmse_train)
    print(diff_mae_test)
    print(diff_rmse_test)

    r2_train_rf = r2_score(y_train, y_pred_train_rf)
    r2_test_rf = r2_score(y_test, y_pred_test_rf)

    # For reduced model
    r2_train_rf_reduced = r2_score(y_train, y_pred_train_rf_reduced)
    r2_test_rf_reduced = r2_score(y_test, y_pred_test_rf_reduced)

    # Add the following line before the return statement in your function
    best_rf_params = best_rf.get_params()

    # Plot learning curve
    from sklearn.model_selection import learning_curve

    def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                            n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                label="Cross-validation score")

        plt.legend(loc="best")
        return plt 

    plot_learning_curve(best_rf, "Learning Curves (Random Forest)", X, y, cv=3)

    # Change the return statement to include the best_rf_params
    return best_rf.get_params(), mae_train_rf, rmse_train_rf, r2_train_rf, mae_test_rf, rmse_test_rf, r2_test_rf, mae_train_rf_reduced, rmse_train_rf_reduced, r2_train_rf_reduced, mae_test_rf_reduced, rmse_test_rf_reduced, r2_test_rf_reduced, total_time

    # return diff_mae_train, diff_rmse_train, diff_mae_test, diff_rmse_test
# %%
rf_out_res_0_5_mil = rf_mod_compare(df, res_var = 'Resistivity_lyrs_9_rad_0_5_miles')
# %%
rf_out_res_1_mil = rf_mod_compare(df, res_var = 'Resistivity_lyrs_9_rad_1_miles')
# %%
rf_out_res_1_5_mil = rf_mod_compare(df, res_var = 'Resistivity_lyrs_9_rad_1_5_miles')
#%%
rf_out_res_2_mil = rf_mod_compare(df, res_var = 'Resistivity_lyrs_9_rad_2_miles')
#%%
rf_out_res_2_5_mil = rf_mod_compare(df, res_var = 'Resistivity_lyrs_9_rad_2_5_miles')
#%%
rf_out_res_3_mil = rf_mod_compare(df, res_var = 'Resistivity_lyrs_9_rad_3_miles')

# %%
# create a list of tuples containing the variable names and values
data = [('0.5', *rf_out_res_0_5_mil),
        ('1', *rf_out_res_1_mil),
        ('1.5', *rf_out_res_1_5_mil),
        ('2', *rf_out_res_2_mil),
        ('2.5', *rf_out_res_2_5_mil),
        ('3', *rf_out_res_3_mil)]

# create a pandas DataFrame from the data
df_diff_error = pd.DataFrame(data, columns=['Radius (miles)', 'Best Params', 'MAE training', 'RMSE training', 'R2 training', 'MAE test', 'RMSE test', 'R2 test', 'MAE training reduced', 'RMSE training reduced', 'R2 training reduced', 'MAE test reduced', 'RMSE test reduced', 'R2 test reduced', 'Training time'])

# # create a pandas DataFrame from the data
# df_diff_error = pd.DataFrame(data, columns=['Radius (miles)', 'MAE diff training', 'RMSE diff training', 'MAE diff test', 'RMSE diff test'])

df_diff_error.to_csv(config.data_processed / 'rfmodel_config/rf_model_outputs.csv', index=False)

# print the DataFrame
print(df_diff_error)

#%%
# Creating table for paper

df_diff_error2 = df_diff_error.copy()
df_diff_error2['MAE training - MAE training reduced'] = df_diff_error['MAE training'] - df_diff_error['MAE training reduced']
df_diff_error2['MAE test - MAE test reduced'] = df_diff_error['MAE test'] - df_diff_error['MAE test reduced']

# First, melt your dataframe to long format
df_long = df_diff_error2.melt(id_vars='Radius (miles)', var_name='Metric', value_name='Value')


# Ensure the data type is numeric
df_long['Value'] = pd.to_numeric(df_long['Value'], errors='coerce')

# Select only rows with metrics you are interested in
metrics = ['MAE training', 'MAE test', 'R2 training', 'R2 test','MAE training - MAE training reduced','MAE test - MAE test reduced']
df_selected = df_long[df_long['Metric'].isin(metrics)]

# Pivot your dataframe to wide format
df_pivot = df_selected.pivot(index='Metric', columns='Radius (miles)', values='Value')

# Round to 2 decimals
df_pivot = df_pivot.round(3)

# Define your order
order = ['MAE training', 'MAE test', 'R2 training', 'R2 test', 'MAE training - MAE training reduced', 'MAE test - MAE test reduced']

# Apply the order to your DataFrame
df_pivot = df_pivot.reindex(order)

# Display your final DataFrame
print(df_pivot)

df_pivot.to_csv('/Users/szalam/Main/00_Research_projects/rf_table_for_manuscript.csv')

#===============================
# Checking overfitting
#===============================
# assuming rf_out_res_0_5_mil is the output from your function
best_rf, mae_train_rf, rmse_train_rf, r2_train_rf, mae_test_rf, rmse_test_rf, r2_test_rf, mae_train_rf_reduced, rmse_train_rf_reduced, r2_train_rf_reduced, mae_test_rf_reduced, rmse_test_rf_reduced, r2_test_rf_reduced, total_time = rf_out_res_2_mil


print("Difference in RMSE between training and test sets: ", rmse_test_rf - rmse_train_rf)
print("Difference in MAE between training and test sets: ", mae_test_rf - mae_train_rf)
#If the difference in RMSE or MAE between training and test sets is significantly positive, it's a sign that your model could be overfitting


#%%

# If  training score is much higher than the cross-validation score and there's a large gap between the curves, it's a sign of overfitting.

# %%
# plot a line graph of radius vs MAE training-test
plt.figure(figsize=(8, 6))
plt.plot(df_diff_error['Radius (miles)'], (df_diff_error['MAE training reduced']-df_diff_error['MAE training']), label='Train', linestyle='--', marker='o')
plt.plot(df_diff_error['Radius (miles)'], (df_diff_error['MAE test reduced']-df_diff_error['MAE test']), label='Test', linestyle='-.', marker='x')

# add labels and title to the plot
plt.xlabel('Radius from well (miles)',size = 14)
plt.ylabel('MAE Δ with/without Resistivity',size = 14)
plt.title('Radius vs MAE difference ',size = 15)

# add a legend to the plot
plt.legend(fontsize = 13)

# change the font size of the x-axis tick labels
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)

# show the plot
plt.show()
# %%
plt.figure(figsize=(8, 6))
# plot a line graph of radius vs MAE training-test
plt.plot(df_diff_error['Radius (miles)'], (df_diff_error['RMSE training reduced']-df_diff_error['RMSE training']), label='Train', linestyle='--', marker='o')
plt.plot(df_diff_error['Radius (miles)'], (df_diff_error['RMSE test reduced']-df_diff_error['RMSE test']), label='Test', linestyle='-.', marker='x')

# add labels and title to the plot
plt.xlabel('Radius from well (miles)', size = 14)
plt.ylabel('RMSE Δ with/without Resistivity)', size = 14)
plt.title('Radius vs RMSE difference', size = 15)

# add a legend to the plot
plt.legend(fontsize = 13)
# change the font size of the x-axis tick labels
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
# show the plot
plt.show()
# %%
plt.figure(figsize=(8, 6))
# plot a line graph of radius vs MAE training-test
plt.plot(df_diff_error['Radius (miles)'], (df_diff_error['R2 training reduced']-df_diff_error['R2 training']), label='Train', linestyle='--', marker='o')
plt.plot(df_diff_error['Radius (miles)'], (df_diff_error['R2 test reduced']-df_diff_error['R2 test']), label='Test', linestyle='-.', marker='x')

# add labels and title to the plot
plt.xlabel('Radius from well (miles)', size = 14)
plt.ylabel('R2 Δ with/without Resistivity)', size = 14)
plt.title('Radius vs R2 difference', size = 15)

# add a legend to the plot
plt.legend(fontsize = 13)
# change the font size of the x-axis tick labels
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
# show the plot
plt.show()

# %%