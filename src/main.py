"""
This script runs a series of Python scripts sequentially to preprocess and postprocess
"""

import subprocess

#============================================================
# Preprocess
#============================================================
# Script: get_nitratedata_gama.py
subprocess.call(["python", "data/get_nitratedata_gama.py"])

# Script: get_nitratedata_ucd.py
subprocess.call(["python", "data/get_nitratedata_ucd.py"])

# Script: get_aem_conductivity_wellbuffer_for_layers.py
subprocess.call(["python", "data/get_aem_conductivity_wellbuffer_for_layers.py"])

# Script: get_aemvalue_wellbuffer_multiregions.py
subprocess.call(["python", "data/get_aemvalue_wellbuffer_multiregions.py"])

# Script: get_cafopop_inbuffer.py
subprocess.call(["python", "data/get_cafopop_inbuffer.py"])

# Script: get_gwdepth_at_wells.py
subprocess.call(["python", "data/get_gwdepth_at_wells.py"])

# Script: get_lucdl.py
subprocess.call(["python", "data/get_lucdl.py"])

# Script: get_lucdl_1.py
subprocess.call(["python", "data/get_lucdl_1.py"])

# Script: get_sagbi_well_buffer.py
subprocess.call(["python", "data/get_sagbi_well_buffer.py"])

# Script: get_subreg_gwpa_for_wells.py
subprocess.call(["python", "data/get_subreg_gwpa_for_wells.py"])

# Script: get_thickness_combined.R
subprocess.call(["Rscript", "data/get_thickness_combined.R"])

# Script: get_thickness_wellbuffer_multiregions.py
subprocess.call(["python", "data/get_thickness_wellbuffer_multiregions.py"])

# Script: get_well_in_city.py
subprocess.call(["python", "data/get_well_in_city.py"])


#============================================================
# Combine all dataset to single csv
#============================================================
# Script: get_aggregate_dataset.py
subprocess.call(["python", "data/get_aggregate_dataset.py"])


#============================================================
# Postprocess
#============================================================

# Depth average resistivity and nitrate relationship
## Resistivity vs nitrate plot
subprocess.call(["python", "analysis/resistivity_nitrate_plot.py"])


# Clay thickness and nitrate relationship
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