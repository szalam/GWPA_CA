
#%%
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, 'src')
import config

#%%
res_var = [
    'Resistivity_lyrs_9_rad_2_miles'
]

#%%

# Constants
rad_well = 2
gama_old_new = 2 # 1: earlier version, 2: latest version of GAMA
all_dom_flag = 1 # 1: All, 2: Domestic
if all_dom_flag == 2:
    well_type_select = {1: 'Domestic', 2: 'DOMESTIC'}.get(gama_old_new) 
else:
    well_type_select = 'All'

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
#%%
df = df.drop(['well_id', 'well_data_source','start_date', 'end_date'], axis=1)

columns_to_keep = ['mean_nitrate','N_total',#'Average_ag_area'
                        'ProbDOpt5ppm_Shallow','ProbDOpt5ppm_Deep','ProbMn50ppb_Shallow','ProbMn50ppb_Deep','PrecipMinusETin_1971_2000_GWRP', #'area_wt_sagbi',
                        'CAML1990_natural_water','DTW60YrJurgens', 'HiWatTabDepMin', 'LateralPosition',  'RechargeAnnualmmWolock', 'RiverDist_NEAR','well_type_encoded',
                        'Resistivity_lyrs_9_rad_2_miles']

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

# %%
filepath = f'/Users/szalam/Main/00_Research_projects/GWPA_rf_outputs/rfmodel_config/rfmodel_parameters_resistivityUse_Resistivity_lyrs_9_rad_2_miles.csv'
df_metric = pd.read_csv(filepath)

# Prepare data
selected_columns = ['rank_test_score', 'mean_train_score', 'mean_test_score']
df_metric[selected_columns] = df_metric[selected_columns].applymap(lambda x: round(x, 4))

# Sort best models
df_sorted = df_metric.sort_values(by=['mean_test_score'], ascending=[False])
    
# %%
# Get top model
top_model = df_sorted.iloc[0]

# %%
# Fit the model using the best parameters
best_params_str = top_model['params']
best_params = ast.literal_eval(best_params_str)
best_model = RandomForestRegressor(**best_params)
# %%
# train the model
best_model.fit(X_train, y_train)

# get feature importances
feature_importances = best_model.feature_importances_
#%%

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
                'Resistivity_lyrs_9_rad_2_miles': 'Depth avg. resistivity (~32m)',
                'Resistivity_lyrs_6': 'Resistivity 6 layer',
                'Resistivity_lyrs_4': 'Resistivity 4 layer'}

X_train = X_train.rename(columns=new_column_names)
#%%
# Get feature names
feature_names = X_train.columns

#%%
import seaborn as sns

def plot_feature_importance(feature_importances, feature_names, title):
    """
    Plots the importance of features from a fitted sklearn model
    """
    # Create a pandas DataFrame for easier plotting
    feature_importances_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})

    # Sort the DataFrame by feature importance
    feature_importances_df.sort_values('importance', ascending=False, inplace=True)

    plt.figure(figsize=(10, 10))
    plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    plt.title(title, fontsize=22)
    sns.barplot(x='importance', y='feature', data=feature_importances_df, orient='h', color='orange',edgecolor='b')
    plt.xlabel('Relative Importance', fontsize=18)
    plt.ylabel('Features', fontsize=20)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.show()


#%%
# Plot feature importance
plot_feature_importance(feature_importances, feature_names, f'Feature Importance')

# %%
# import shap

# # create the explainer object with the random forest model
# explainer = shap.Explainer(best_model)

# # Transform the test set
# shap_values = explainer.shap_values(X_test)

# #%%
# # plot
# shap.summary_plot(shap_values, X_test, plot_type="bar")
# shap.dependence_plot('Resistivity_lyrs_9_rad_2_miles', shap_values, X_test)

# %%
