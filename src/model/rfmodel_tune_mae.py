#%%
import sys
sys.path.insert(0,'src')
import matplotlib.pyplot as plt
import numpy as np
import config
import pandas as pd

from tqdm import tqdm
from time import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

#%%
# Constants
rad_well = 2
gama_old_new = 2 # 1: earlier version, 2: latest version of GAMA
all_dom_flag = 1 # 1: All, 2: Domestic
if all_dom_flag == 2:
    well_type_select = {1: 'Domestic', 2: 'DOMESTIC'}.get(gama_old_new) 
else:
    well_type_select = 'All'


# Function definitions 
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

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    metrics = {
        'mae_train': mean_absolute_error(y_train, y_pred_train),
        'rmse_train': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'r2_train': r2_score(y_train, y_pred_train),
        'mae_test': mean_absolute_error(y_test, y_pred_test),
        'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'r2_test': r2_score(y_test, y_pred_test),
    }
    
    return model, metrics


def tune_and_evaluate(estimator, params, X_train, y_train, X_test, y_test):
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=params,
        cv=3,
        scoring='neg_mean_squared_error',
        return_train_score=True,
        verbose=3,
        n_jobs=-1
    )

    start_time = time()
    grid_search.fit(X_train, y_train)
    total_time = time() - start_time

    best_model, metrics = train_and_evaluate(grid_search.best_estimator_, X_train, y_train, X_test, y_test)
    metrics['time'] = total_time

    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results['mae_cv'] = -cv_results['mean_test_score']
    cv_results['r2_cv'] = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=3, scoring='r2').mean()

    return best_model, metrics, cv_results

def rf_mod_compare(df, res_var='Resistivity_lyrs_9_rad_2_miles'):
    columns_to_keep = ['mean_nitrate','N_total',#'Average_ag_area'
                        'ProbDOpt5ppm_Shallow','ProbDOpt5ppm_Deep','ProbMn50ppb_Shallow','ProbMn50ppb_Deep',
                        'PrecipMinusETin_1971_2000_GWRP', #'area_wt_sagbi',
                        'CAML1990_natural_water','DTW60YrJurgens', 'HiWatTabDepMin', 'LateralPosition',  
                        'RechargeAnnualmmWolock', 'RiverDist_NEAR','well_type_encoded',
                        f'{res_var}']
    
    df2 = df[columns_to_keep]
    df2 = df2.dropna()

    # Features
    X = df2.drop("mean_nitrate", axis=1)

    # Target
    y = df2["mean_nitrate"]
    y[y<=0]=.00001
    y = np.log(y)

    # Split the data into a training set and a testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_params = {
        'n_estimators': [500, 5000, 10000],
        'max_depth': [5, 25, 50, None],
        'min_samples_split': [2, 50, 100, 500],
        'min_samples_leaf': [5, 50, 100, 500],
        'max_features' : ['sqrt', 0.25, 0.5],
    }

    rf = RandomForestRegressor(n_estimators=100, random_state=42,bootstrap=True)

    best_rf, metrics_full, cv_results = tune_and_evaluate(rf, rf_params, X_train, y_train, X_test, y_test)

    X_train_reduced = X_train.drop([f'{res_var}'], axis=1)
    X_test_reduced = X_test.drop([f'{res_var}'], axis=1)

    best_rf_reduced, metrics_reduced, cv_results_reduced = tune_and_evaluate(rf, rf_params, X_train_reduced, y_train, X_test_reduced, y_test)
    
    return best_rf.get_params(), metrics_full, best_rf_reduced.get_params(), metrics_reduced, cv_results, cv_results_reduced

# Plot learning curve
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

def get_xy_train_test(df):
    columns_to_keep = ['mean_nitrate','N_total',#'Average_ag_area'
                        'ProbDOpt5ppm_Shallow','ProbDOpt5ppm_Deep','ProbMn50ppb_Shallow','ProbMn50ppb_Deep',
                        'PrecipMinusETin_1971_2000_GWRP', #'area_wt_sagbi',
                        'CAML1990_natural_water','DTW60YrJurgens', 'HiWatTabDepMin', 'LateralPosition',  
                        'RechargeAnnualmmWolock', 'RiverDist_NEAR','well_type_encoded',
                        f'{res_var}']
    
    df2 = df[columns_to_keep]
    df2 = df2.dropna()

    # Features
    X = df2.drop("mean_nitrate", axis=1)

    # Target
    y = df2["mean_nitrate"]
    y[y<=0]=.00001
    y = np.log(y)

    # Split the data into a training set and a testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
#%%
# Main Code

# Load and process data
df_main = load_data(gama_old_new)
df = df_main[df_main.well_data_source == 'GAMA'].copy()

df['well_type_encoded'] = pd.factorize(df['well_type'])[0]
df['well_type_encoded'] = df['well_type_encoded'].where(df['well_type'].notna(), df['well_type'])

# separate wells inside cv
well_cv = pd.read_csv(config.data_processed / 'wells_inside_CV_GAMAlatest.csv',index_col=False)

# Assuming df is your dataframe with all wells
df_cv = df[df['well_id'].isin(well_cv['well_id'])]
df = filter_data(df_cv, well_type_select,all_dom_flag)
df = df.drop(['well_id', 'well_data_source','start_date', 'end_date'], axis=1)

#%%
# For the first loop:
for i, res_var in tqdm(enumerate(['Resistivity_lyrs_9_rad_2_miles']), total=1, desc="Running models", unit="model"):
    best_params, metrics_full, best_params_reduced, metrics_reduced, cv_results, cv_results_reduced = rf_mod_compare(df, res_var=res_var)
    cv_results.to_csv(f'/Users/szalam/Main/00_Research_projects/GWPA_rf_outputs/rfmodel_config/rfmodel_parameters_resistivityUse_{res_var}.csv', index=False)
    cv_results_reduced.to_csv(f'/Users/szalam/Main/00_Research_projects/GWPA_rf_outputs/rfmodel_config/rfmodel_parameters_resistivityUse_{res_var}_reduced.csv', index=False)

#%%
X_train, X_test, y_train, y_test = get_xy_train_test(df)
# create model with best parameters
best_model = RandomForestRegressor(**best_params)

# train the model
best_model.fit(X_train, y_train)

# get feature importances
feature_importances = best_model.feature_importances_
#%%
# Get feature names
feature_names = X_train.columns

#%%
def plot_feature_importance(feature_importances, feature_names, title):
    """
    Plots the importance of features from a fitted sklearn model
    """
    indices = np.argsort(feature_importances)

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.barh(range(len(indices)), feature_importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

#%%
# Plot feature importance
plot_feature_importance(feature_importances, feature_names, f'Feature Importance with {res_var}')

#%%
import shap

# create the explainer object with the random forest model
explainer = shap.Explainer(best_model)

# Transform the test set
shap_values = explainer.shap_values(X_test)

# plot
shap.summary_plot(shap_values, X_train, plot_type="bar")
shap.dependence_plot('Resistivity_lyrs_9_rad_2_miles', shap_values, X_test)
