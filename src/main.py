"""
This script runs a series of Python scripts sequentially to preprocess and postprocess
"""

import subprocess

#============================================================
# Preprocess
#============================================================
# Preprocess latest gama dataset
subprocess.call(["python", "data/gama_latestdata_combine.py"])

# Get AEM values at different well buffers
subprocess.call(["python", "data/get_aemvalue_wellbuffer_multiregions.py"])

# Calculate the thickness of clay layers for different resistivity threshold levels and well radius
subprocess.call(["python", "data/get_thickness_wellbuffer_multiregions.py"])

# Combine the thickness of aem values to single csv. This is a R code. I used Rstudio for running the following
# get_thickness_combined.R

# The data from previous literature Ransom et al. was extracted using code ran in colab
# The code first required separating GAMA well locations to be imported by colab code
subprocess.call(["python", "data/get_gama_wells_csv_locations.py"]) # separate well locations
# get_redox_data.ipynb # colab code that has been run


# Script: get_lucdl.py
subprocess.call(["python", "data/get_lucdl.py"])

# Script: get_lucdl_1.py
subprocess.call(["python", "data/get_lucdl_1.py"])

# Script: get_subreg_gwpa_for_wells.py
subprocess.call(["python", "data/get_subreg_gwpa_for_wells.py"])


#============================================================
# Combine all dataset to single csv
#============================================================
# Script: get_aggregate_dataset.py
subprocess.call(["python", "data/get_aggregate_dataset_GAMAlatest.py"])

#============================================================
# Postprocess
#============================================================

# Depth average resistivity and nitrate relationship
## Resistivity vs nitrate plot
subprocess.call(["python", "analysis/resistivity_nitrate_plot.py"])

## Clay thickness and nitrate box plot
subprocess.call(["python", "analysis/clay_thickness_nitrate.py"])

## Clay thickness and nitrate heatmap
subprocess.call(["python", "analysis/clay_thickness_nitrate_bivariate.py"])

# Nitrate and resistivity distribution in all wells vs inside leaching GWPA
subprocess.call(["python", "analysis/nitrate_resistivity_inout_gwpa_boxplot.py"])

#============================================================
# Random forest
#============================================================

# Random forest model hyper parameter tuning and error estimation.
subprocess.call(["python", "model/rfmodel_tune_mae.py"])
# Random forest model performance evaluation
subprocess.call(["python", "model/rfmodel_performance_evaluation.py"])
# Random forest model relative importance determination.
subprocess.call(["python", "model/rfmodel_feature_importance.py"])