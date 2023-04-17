#================================================================================
# This script analyzies the relationship between resistivity and nitrate for 
# different levels of redox condition and N inputs
#================================================================================
#%%
import sys
sys.path.insert(0,'src')
sys.path.insert(0, 'src/data')

import pandas as pd
import config
import numpy as np
import matplotlib.pyplot as plt 
import ppfun as dp
from scipy import stats
import seaborn as sns
import matplotlib as mpl

df_main = pd.read_csv(config.data_processed / "Dataset_processed.csv")
#%%
lyrs = 9
rad_well = 2
cond_type_used = 'Resistivity'

if cond_type_used == 'Conductivity':
    aem_type = 'Conductivity'
else:
    aem_type = 'Resistivity'


def filter_dataframe(df, well_data_source=None, measurement_count=None, city_inside_outside=None, subregion=None,
                      well_type=None, crop_threshold=None, exclude_saline_subregions=None, min_mean_nitrate=None):
    
    if well_data_source:
        df = df[df.well_data_source == well_data_source]
    if measurement_count:
        df = df[df.measurement_count > measurement_count]
    if city_inside_outside:
        df = df[df.city_inside_outside == city_inside_outside]
    if subregion:
        df = df[df['SubRegion'] == subregion]
    if well_type:
        df = df[df.well_type == well_type] # 'Water Supply, Other', 'Municipal', 'Domestic'
    if crop_threshold:
        df = df[df.All_crop_2015 >= crop_threshold]
    if exclude_saline_subregions:
        exclude_subregions = [14, 15, 10, 19,18, 9,6]
        df = df[(df[f'thickness_abovCond_{round(.1*100)}_lyrs_9_rad_{rad_well}miles'] <= 31) | (~df['SubRegion'].isin(exclude_subregions))]
    if min_mean_nitrate:
        df = df[df.mean_nitrate > min_mean_nitrate]
        
    return df

df = filter_dataframe(df_main, well_data_source='GAMA', subregion= 15, exclude_saline_subregions = True)

# %%
if aem_type == 'Resistivity':
    df = df[df[f'{cond_type_used}_lyrs_{lyrs}']<100]

# Bin Conductivity into intervals of .05
if aem_type == 'Conductivity':
    df = df[df[f'{cond_type_used}']<1]
    df['Conductivity_binned'] = pd.cut(df[f'{cond_type_used}_lyrs_{lyrs}'], 
                                       np.arange(0, 0.2,.01))
if aem_type == 'Resistivity':
    df['Conductivity_binned'] = pd.cut(df[f'{cond_type_used}_lyrs_{lyrs}'], 
                                       np.arange(0, df[f'{cond_type_used}_lyrs_{lyrs}'].max()+1,5))


#%%
# Create the scatter plot
plt.scatter(df["ProbMn50ppb_Shallow"], df["mean_nitrate"])
plt.xlabel("ProbMn50ppb_Shallow")
plt.ylabel("mean_nitrate")
plt.ylim(0,250)
plt.title("Scatter Plot of ProbMn50ppb_Shallow vs mean_nitrate")
plt.show()

#%%
# Create the scatter plot
plt.scatter(df["ProbMn50ppb_Deep"], df["mean_nitrate"])
plt.xlabel("ProbMn50ppb_Deep")
plt.ylabel("mean_nitrate")
plt.ylim(0,250)
plt.title("Scatter Plot of ProbMn50ppb_Deep vs mean_nitrate")
plt.show()

#%%
# Create the scatter plot
plt.scatter(df["All_crop_2008"], df["mean_nitrate"])
plt.xlabel("Total crop area 2008")
plt.ylim(0,400)
plt.ylabel("mean_nitrate")
plt.title("Scatter Plot of total crop in 2008 vs mean_nitrate")
plt.show()

#%%
# Create the scatter plot
plt.scatter(df["N_total"], df["mean_nitrate"])
plt.xlabel("N input")
plt.ylabel("mean_nitrate")
plt.ylim(0,250)
plt.title("Scatter Plot of Nitrogen input vs mean_nitrate")
plt.show()
#%%
# Create the scatter plot
plt.scatter(df["ProbDOpt5ppm_Shallow"], df["mean_nitrate"])
plt.xlabel("ProbDOpt5ppm_Shallow")
plt.ylabel("mean_nitrate")
plt.ylim(0,250)
plt.title("Scatter Plot of ProbDOpt5ppm_Shallow vs mean_nitrate")
plt.show()

#%%
# Create the scatter plot
plt.scatter(df[f'{cond_type_used}_lyrs_{lyrs}'], df["ProbMn50ppb_Shallow"])
plt.xlabel("Resistivity")
plt.ylabel("Prob of Mn/DO")
plt.show()


#%%
prob_threshold = .1
df_Mn_l = df[df.ProbMn50ppb_Shallow<prob_threshold]  
df_Mn_h = df[df.ProbMn50ppb_Shallow>prob_threshold]  

def plot_box_and_whisker(df, aem_type):
    sns.boxplot(x='Conductivity_binned', y='mean_nitrate', data=df, width=0.5, color = 'orange')
    
    plt.xlabel(f'Depth Average {aem_type} (\u2126-m)', fontsize = 13)
    plt.ylabel('Nitrate-N [mg/l]', fontsize =13)
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.yscale('log')
    plt.ylim(0.01, 1500)

    plt.xticks(rotation=90)

    plt.show()

plot_box_and_whisker(df_Mn_l,aem_type)
plot_box_and_whisker(df_Mn_h,aem_type)
plot_box_and_whisker(df,aem_type)
#%%
# prob_threshold = .2
# df_DO_l = df[df.ProbDOpt5ppm_Shallow<prob_threshold]  
# df_DO_h = df[df.ProbDOpt5ppm_Shallow>prob_threshold]  

# def plot_box_and_whisker(df, aem_type):
#     sns.boxplot(x='Conductivity_binned', y='mean_nitrate', data=df, width=0.5, color = 'orange')
    
#     plt.xlabel(f'Depth Average {aem_type} (\u2126-m)', fontsize = 13)
#     plt.ylabel('Nitrate-N [mg/l]', fontsize =13)
#     plt.tick_params(axis='both', which='major', labelsize=10)

#     plt.yscale('log')
#     plt.ylim(0.01, 1500)

#     plt.xticks(rotation=90)

#     plt.show()

# plot_box_and_whisker(df_DO_l,aem_type)
# plot_box_and_whisker(df_DO_h,aem_type)

#%%
n_thresh = 9000
df_Nin_l = df[df.N_total<n_thresh]  
df_Nin_h = df[df.N_total>n_thresh]  

def plot_box_and_whisker(df, aem_type):
    sns.boxplot(x='Conductivity_binned', y='mean_nitrate', data=df, width=.6, color = 'orange')
    
    plt.xlabel(f'Depth Average {aem_type} (\u2126-m)', fontsize = 13)
    plt.ylabel('Nitrate-N [mg/l]', fontsize =13)
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.yscale('log')
    plt.ylim(0.01, 1500)

    plt.xticks(rotation=90)

    plt.show()

plot_box_and_whisker(df_Nin_l,aem_type)
plot_box_and_whisker(df_Nin_h,aem_type)

#%%
n_thresh = 5000
df_Mn_sel = df_Mn_l.copy()
df_Nin_l_2 = df_Mn_sel[df_Mn_sel.N_total<n_thresh]  
df_Nin_h_2 = df_Mn_sel[df_Mn_sel.N_total>n_thresh]  

def plot_box_and_whisker(df, aem_type):
    sns.boxplot(x='Conductivity_binned', y='mean_nitrate', data=df, width=.6, color = 'orange')
    
    plt.xlabel(f'Depth Average {aem_type} (\u2126-m)', fontsize = 13)
    plt.ylabel('Nitrate-N [mg/l]', fontsize =13)
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.yscale('log')
    plt.ylim(0.01, 1500)

    plt.xticks(rotation=90)

    plt.show()

plot_box_and_whisker(df_Nin_l_2,aem_type)
plot_box_and_whisker(df_Nin_h_2,aem_type)
#%%
n_thresh = .1
df_Nin_sel = df_Nin_h.copy()
df_Mn_l_2 = df_Nin_sel[df_Nin_sel.ProbMn50ppb_Shallow<n_thresh]  
df_Mn_h_2 = df_Nin_sel[df_Nin_sel.ProbMn50ppb_Shallow>n_thresh]  

def plot_box_and_whisker(df, aem_type):
    sns.boxplot(x='Conductivity_binned', y='mean_nitrate', data=df, width=.6, color = 'orange')
    
    plt.xlabel(f'Depth Average {aem_type} (\u2126-m)', fontsize = 13)
    plt.ylabel('Nitrate-N [mg/l]', fontsize =13)
    plt.tick_params(axis='both', which='major', labelsize=10)

    plt.yscale('log')
    plt.ylim(0.01, 1500)

    plt.xticks(rotation=90)

    plt.show()

plot_box_and_whisker(df_Mn_l_2,aem_type)
plot_box_and_whisker(df_Mn_h_2,aem_type)

# %%
# Create intervals for x-axis
bins = np.arange(0, 1.2, 0.1)
labels = [f"{x:.1f}-{x+0.2:.1f}" for x in bins[:-1]]

# Assign each data point to a bin
df['Mn_shallow_bin'] = pd.cut(df['ProbMn50ppb_Shallow'], bins, labels=labels, include_lowest=True)

# Group by bin and create a new dataframe with the grouped data
grouped_df = df.groupby('Mn_shallow_bin')['mean_nitrate'].apply(list).reset_index()

# Create the box and whisker plot
sns.boxplot(x='Mn_shallow_bin', y='mean_nitrate', data=df, width=.6, color = 'orange')
    
plt.xlabel(f'Mn probability in shallow aquifer', fontsize = 13)
plt.ylabel('Nitrate-N [mg/l]', fontsize =13)
plt.tick_params(axis='both', which='major', labelsize=10)

plt.yscale('log')
plt.ylim(0.01, 1500)

plt.xticks(rotation=90)

plt.show()

# %%
