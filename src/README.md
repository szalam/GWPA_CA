## Description
This repository contains a comprehensive script for water quality dataset analysis and investigation of its relationship with resistivity and other relevant physical properties. The code is designed to handle extensive data preprocessing tasks to ensure accurate and reliable results. Subsequently, the processed data is utilized for in-depth analysis and the development of a Random Forest model, a powerful machine learning algorithm. This script provides a systematic approach to effectively explore and understand the intricate relationships within the dataset, enabling future users to gain valuable insights and make informed decisions based on the findings.

## Table of Contents
- [Data](#data)
- [Analysis](#analysis)
- [Model](#model)
- [Visualization](#visualization)

## Tree Structure
├── data
│   ├── get_aem_conductivity_wellbuffer_for_layers.py
│   ├── get_aemvalue_wellbuffer_multiregions.py
│   ├── get_aggregate_dataset.py
│   ├── get_cafopop_inbuffer.py
│   ├── get_gwdepth_at_wells.py
│   ├── get_infodata.py
│   ├── get_kml.py
│   ├── get_latlon_nit_csv.py
│   ├── get_lucdl.py
│   ├── get_lucdl_1.py
│   ├── get_nitratedata_gama.py
│   ├── get_nitratedata_ucd.py
│   ├── get_sagbi_well_buffer.py
│   ├── get_subreg_gwpa_for_wells.py
│   ├── get_thickness_combined.R
│   ├── get_thickness_wellbuffer_multiregions.py
│   ├── get_well_in_city.py
│   ├── ppfun.py
│   ├── tmp.py
│   ├── wlevel.py
│   ├── wlfun.py
│   └── wlob.py
├── analysis
│   ├── anomaly_conduc_nitrate_analys.py
│   ├── clay_thickness_nitrate.py
│   ├── clay_thickness_nitrate_bivariate.py
│   ├── conduct_sagbi_plot.py
│   ├── conductivity_spatial.py
│   ├── conus_nitrate_study_explore.py
│   ├── get_conductivity_distribution.py
│   ├── nitplots.py
│   ├── nitrate_belowabove_thresholdResistivity_boxplot.py
│   ├── nitrate_hist_sagbiInterval.py
│   ├── nitrate_res_truefalse.py
│   ├── nitrate_resistivity_inout_gwpa_boxplot.py
│   ├── nitrate_tseries_cafo_dating_compare.py
│   ├── nitrate_tseries_plot_around_CAFO_in_select_wells.py
│   ├── percent_well_protected_by_dpr_gwpa.py
│   ├── redoxdata_well_check_overlaying_map.py
│   ├── resistivity_in_leachGWPA.py
│   ├── resistivity_nitrate_plot.py
│   ├── resistivity_no3_highlow_redox_Ninput.py
│   ├── resistivity_no3_highlowcrop_cafo_boxplot
│   ├── thickness_spatial.py
│   └── visualize.py
├── model
│   ├── kmean.py
│   ├── rfmodel_feature_imptnc.py
│   ├── rfmodel_tune_mae.py
│   ├── scenariotest.py
│   ├── vulnerability_scoring.py
│   └── xgboost.py
└── visualization
│   ├── nitrate_zipcode.R
│   ├── nitrate_zipcode.py
│   └── study_area_map.Rmd
├── config.py
├── utils.py
├── main.py