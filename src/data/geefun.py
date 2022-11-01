# convert geodataframe to ee object
import ee
import ee
import sys
import numpy as np
import pandas as pd
import geopandas as gp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tqdm as tqdm
from pandas.tseries.offsets import MonthEnd

startDate = '2015-01-01'
endDate = '2018-12-31'

def data_info():

	'''
	Information of GEE data sources
	Args: 
	(1) ImageColection
	(2) variable name
	(3) scale factor - needed to calculate volumes when computing sums. Depends on units and sampling frequency 
	(4) native resolution - needed to return gridded images 
	'''
	data = {}

	###################
	##### ET #####
	###################

	# https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD16A2
	data['modis_aet'] = [ee.ImageCollection('MODIS/006/MOD16A2'), "ET", 0.1, 1000]
	data['modis_pet'] = [ee.ImageCollection('MODIS/006/MOD16A2'), "PET", 0.1, 1000]

	# https://developers.google.com/earth-engine/datasets/catalog/NASA_GLDAS_V021_NOAH_G025_T3H
	data['gldas_aet'] = [ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H'), 'Evap_tavg', 86400*30 / 240 , 25000]   # kg/m2/s --> km3 / mon , noting 3 hrly images
	data['gldas_pet'] = [ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H'), 'PotEvap_tavg', 1 / 240, 25000] 

	# https://developers.google.com/earth-engine/datasets/catalog/NASA_NLDAS_FORA0125_H002
	data['nldas_pet'] = [ee.ImageCollection('NASA/NLDAS/FORA0125_H002'), 'potential_evaporation', 1, 12500]

	# https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_TERRACLIMATE
	data['tc_aet'] = [ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE'), "aet", 0.1 , 1000]
	data['tc_pet'] = [ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE'), "pet", 0.1, 1000]

	# https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_GRIDMET
	data['gmet_etr'] = [ee.ImageCollection('IDAHO_EPSCOR/GRIDMET'), "etr", 1 , 1000]
	data['gmet_eto'] = [ee.ImageCollection('IDAHO_EPSCOR/GRIDMET'), "eto", 1, 1000]

	# https://developers.google.com/earth-engine/datasets/catalog/NASA_FLDAS_NOAH01_C_GL_M_V001
	data['fldas_aet'] = [ee.ImageCollection('NASA/FLDAS/NOAH01/C/GL/M/V001'), "Evap_tavg", 86400*30, 12500]

	###################
	##### P data ######
	###################

	data['trmm']  =  [ee.ImageCollection('TRMM/3B43V7'), "precipitation", 720, 25000] # scale hours per month
	data['prism'] = [ee.ImageCollection("OREGONSTATE/PRISM/AN81m"), "ppt", 1, 4000]
	data['chirps'] = [ee.ImageCollection('UCSB-CHG/CHIRPS/PENTAD'), "precipitation", 1, 5500]
	data['persiann'] = [ee.ImageCollection("NOAA/PERSIANN-CDR"), "precipitation", 1, 25000]
	data['dmet'] = [ee.ImageCollection('NASA/ORNL/DAYMET_V4'), "prcp", 1, 4000]
	data['gpm'] = [ee.ImageCollection("NASA/GPM_L3/IMERG_MONTHLY_V06"), "precipitation", 720, 12500] # scale hours per month

	#################### 
	##### SWE data #####
	####################
	data['fldas_swe'] = [ee.ImageCollection('NASA/FLDAS/NOAH01/C/GL/M/V001'), "SWE_inst", 1 , 12500]
	data['gldas_swe'] = [ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H'), "SWE_inst", 1 , 25000]
	data['dmet_swe'] = [ee.ImageCollection('NASA/ORNL/DAYMET_V4'), "swe", 1, 4000] # Reduced from 1000 because the query times out over the whole CVW 
	data['tc_swe'] = [ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE'), "swe", 1, 4000]

	####################
	##### R data #######
	####################
	data['tc_r'] = [ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE'), "ro", 1, 4000]
	
	# FlDAS
	data['fldas_ssr'] = [ee.ImageCollection("NASA/FLDAS/NOAH01/C/GL/M/V001"), "Qs_tavg", 86400*30, 12500] # kg/m2/s --> km3 / mon
	data['fldas_bfr'] = [ee.ImageCollection("NASA/FLDAS/NOAH01/C/GL/M/V001"), "Qsb_tavg", 86400*30, 12500]

	# GLDAS
	data['gldas_ssr'] = [ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H'), "Qs_acc", 1, 25000]
	data['gldas_bfr'] = [ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H'), "Qsb_acc", 1, 25000 ]
	data['gldas_qsm'] = [ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H'), "Qsm_acc", 1, 25000]

	# ECMWF
	data['ecmwf_r'] = [ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY") , 'runoff', 1, 10000]
	#####################
	##### SM data #######
	#####################
	data['tc_sm'] = [ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE'), "soil", 0.1, 4000]

	data['fsm1'] = [ee.ImageCollection("NASA/FLDAS/NOAH01/C/GL/M/V001"), "SoilMoi00_10cm_tavg", 86400*24 , 12500]
	data['fsm2'] = [ee.ImageCollection("NASA/FLDAS/NOAH01/C/GL/M/V001"), "SoilMoi10_40cm_tavg", 86400*24 , 12500]
	data['fsm3'] = [ee.ImageCollection("NASA/FLDAS/NOAH01/C/GL/M/V001"), "SoilMoi40_100cm_tavg", 86400*24 , 12500]
	data['fsm4'] = [ee.ImageCollection("NASA/FLDAS/NOAH01/C/GL/M/V001"), "SoilMoi100_200cm_tavg", 86400*24 , 12500]

	data['gldas_rzsm'] = [ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H'), "RootMoist_inst", 1, 25000]

	data['gsm1'] = [ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H"), "SoilMoi0_10cm_inst", 1 ,25000]
	data['gsm2'] = [ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H"), "SoilMoi10_40cm_inst", 1 ,25000]
	data['gsm3'] = [ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H"), "SoilMoi40_100cm_inst", 1 ,25000]
	data['gsm4'] = [ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H"), "SoilMoi100_200cm_inst", 1 ,25000]

	data['smap_ssm'] = [ee.ImageCollection("NASA_USDA/HSL/SMAP_soil_moisture"), "ssm", 1 ,25000]
	data['smap_susm'] = [ee.ImageCollection("NASA_USDA/HSL/SMAP_soil_moisture"), "susm", 1 ,25000]
	data['smap_smp'] = [ee.ImageCollection("NASA_USDA/HSL/SMAP_soil_moisture"), "smp", 1 ,25000]

	data['smos_ssm'] = [ee.ImageCollection("NASA_USDA/HSL/soil_moisture"), "ssm", 1 ,25000]
	data['smos_susm'] = [ee.ImageCollection("NASA_USDA/HSL/soil_moisture"), "susm", 1 ,25000]
	data['smos_smp'] = [ee.ImageCollection("NASA_USDA/HSL/soil_moisture"), "smp", 1 ,25000]
	############################
	##### Elevation #######
	############################

	data['srtm'] = [ee.Image("CGIAR/SRTM90_V4"), "elevation", 1 ,1000]

	#########################
	##### Gravity data ######
	#########################
	data['jpl'] = [ee.ImageCollection('NASA/GRACE/MASS_GRIDS/LAND'), "lwe_thickness_jpl",  ee.Image("NASA/GRACE/MASS_GRIDS/LAND_AUX_2014").select("SCALE_FACTOR"), 25000]
	data['csr'] = [ee.ImageCollection('NASA/GRACE/MASS_GRIDS/LAND'), "lwe_thickness_csr",  ee.Image("NASA/GRACE/MASS_GRIDS/LAND_AUX_2014").select("SCALE_FACTOR"), 25000]
	data['gfz'] = [ee.ImageCollection('NASA/GRACE/MASS_GRIDS/LAND'), "lwe_thickness_gfz",  ee.Image("NASA/GRACE/MASS_GRIDS/LAND_AUX_2014").select("SCALE_FACTOR"), 25000]

	data['mas'] = [ee.ImageCollection('NASA/GRACE/MASS_GRIDS/MASCON'), "lwe_thickness", 1] 
	data['mas_unc'] = [ee.ImageCollection('NASA/GRACE/MASS_GRIDS/MASCON'), "uncertainty", 1] 

	data['cri'] = [ee.ImageCollection('NASA/GRACE/MASS_GRIDS/MASCON_CRI'), "lwe_thickness", 1] 
	data['cri_unc'] = [ee.ImageCollection('NASA/GRACE/MASS_GRIDS/MASCON_CRI'), "uncerrtainty", 1] 


	#########################
	##### Optical ######
	#########################

	data['modis_snow'] = [ee.ImageCollection('MODIS/006/MOD10A1'), "NDSI_Snow_Cover",  1, 2500] # reduced resolution  
	data['modis_ndvi'] = [ee.ImageCollection('MODIS/MOD09GA_006_NDVI'), "NDVI",  1, 250] 

	data['landsat_8_b1'] = [ee.ImageCollection('LANDSAT/LC08/C01/T1_SR'), "B1" ,  0.001, 30] 

	data['l8_ndwi_32d'] = [ee.ImageCollection('LANDSAT/LC08/C01/T1_32DAY_NDWI'), "NDWI", 1, 30]
	data['l8_ndwi_annual'] = [ee.ImageCollection("LANDSAT/LC08/C01/T1_ANNUAL_NDWI"), "NDWI", 1, 30]


	###########################
	##### Landcover ######
	###########################
	data['cdl'] = [ee.ImageCollection('USDA/NASS/CDL'), "cropland",  1, 30]
	data['nlcd'] = [ee.ImageCollection('USGS/NLCD'), 'landcover', 1, 30]

	return data


def gp_to_ee_poly(gdf, simplify = False):
    """
    Geopandas dataframe to EE polygon
    Arg:
    gdf: geopandas dataframe
    simplify: simplifying the polygon
    """
    if simplify:
        gdf = gdf.geometry.simplify(0.01)

    lls = gdf.geometry.iloc[0]
    x,y = lls.exterior.coords.xy
    coords = [list(zip(x,y))]
    area = ee.Geometry.Polygon(coords)

    return area



def diff_month(d1, d2):
    """
    Difference in months between two dates
    """
    return (d1.year - d2.year) * 12 + d1.month - d2.month



def get_ts_plot(ndvi_df,label_n = 'NDVI'):
    """
    Plot time series
    """
    ndvi_df.columns = ['Date','Value']
    fig, ax = plt.subplots()
    ax.plot(ndvi_df.Date, ndvi_df.Value, linestyle = '--', marker = 'o', label = label_n)

    # ax.xaxis.set_major_locator(mdates.MonthLocator(interval = 1))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.tick_params(axis='x', labelrotation=45)
    ax.legend()



def get_monthly_mean(dataset, startdate, enddate, area):
	'''
	Calculates monthly mean of selected variable

    Note: 
    - There is a multiplying 'mult_factor' may be necessary for unit conversion. default
    value kept 1e-12 which is mm --> km^3
    - This function is relatively slow. a faster option is 'get_reg_ee()'
	'''
	ImageCollection = dataset[0]
	var = dataset[1]
	scaling_factor = dataset[2]
	mult_factor = 1e-12
    
	dt_idx = pd.date_range(startdate,enddate, freq='MS')
	sums = []
	seq = ee.List.sequence(0, len(dt_idx)-1)
	num_steps = seq.getInfo()

	print("processing:")
	print("{}".format(ImageCollection.first().getInfo()['id']))

	for i in num_steps:

		start = ee.Date(startdate).advance(i, 'month')
		end = start.advance(1, 'month');

		im = ee.ImageCollection(ImageCollection).select(var).filterDate(start, end).mean().set('system:time_start', start.millis())
		scale = im.projection().nominalScale()
		scaled_im = im.multiply(scaling_factor).multiply(ee.Image.pixelArea()).multiply(mult_factor) 
		
		sumdict  = scaled_im.reduceRegion(
			reducer = ee.Reducer.sum(),
			geometry = area,
			scale = scale,
			bestEffort = True)

		total = sumdict.getInfo()[var]
		sums.append(total)
		
	sumdf = pd.DataFrame(np.array(sums), dt_idx + MonthEnd(0))
	sumdf.columns = [var]
	df = sumdf.astype(float)
		
	return df



def get_reg_ee(n):
    date = ee.Date(startDate).advance(n,'month')
    m = date.get("month")
    y = date.get("year")
    dic = ee.Dictionary({
        'Date':date.format('yyyy-MM')
    })
    
    tempNDVI = (eedata.filter(ee.Filter.calendarRange(y, y, 'year'))
                .filter(ee.Filter.calendarRange(m, m, 'month'))
                .mean()
                .reduceRegion(
                    reducer = ee.Reducer.mean(),
                    geometry = reg,
                    scale = eescale))
    return dic.combine(tempNDVI)


###############################################
## Processing crop land use data ##############
###############################################
# function to get unique values
def unique(list1):
  
    # initialize a null list
    unique_list = []
      
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    # print list
    return(unique_list)

# count repeat of a selected value
def countX(lst, x):
    count = 0
    for ele in lst:
        if (ele == x):
            count = count + 1
    return count

def cdl_pr(yr,area):
    im = (
    ee.ImageCollection('USDA/NASS/CDL')
    # ee.ImageCollection('USDA/NASS/CDL/{}'.format(str(yr)))
    .filterBounds(area)
    .filterDate((str(yr) +'-01-01'), (str(yr + 1) +'-12-31'))
    .first()
    )

    results_dict  = im.reduceRegion(
        reducer = ee.Reducer.toList(),
        geometry = area,
        scale = 30,
        bestEffort= True)

    cdl_dict = results_dict.getInfo()
    
    return cdl_dict

def nlcd_pr(yr):
    im = (
    ee.ImageCollection('USGS/NLCD_RELEASES/2019_REL/NLCD')
    # ee.ImageCollection('USDA/NASS/CDL/{}'.format(str(yr)))
    .filterBounds(area)
    .filterDate((str(yr) +'-01-01'), (str(yr + 1) +'-12-31'))
    .first()
    )

    results_dict  = im.reduceRegion(
        reducer = ee.Reducer.toList(),
        geometry = area,
        scale = 30,
        bestEffort= True)

    cdl_dict = results_dict.getInfo()
    
    return cdl_dict


def crop_all_area(crop_all,cdl_dict, s_grid, attrb = 'cropland'):
    crop_type = []
    crop_count = []
    for i in crop_all:
        # print('Processing crop: ', str(i))
        #count number of cells with specific crop
        cel_count = countX(cdl_dict[str(attrb)], i)

        #append crop area (km2) and type. /1000000 to convert m2 to km2
        crop_count.append(cel_count * s_grid * s_grid/1000000)
        crop_type.append(i)
    
    df_crp = pd.DataFrame([crop_type,crop_count])
    
    return df_crp

# Note: data_info() function was adapted from Kashington repo

