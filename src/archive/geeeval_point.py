#%%
from cProfile import label
from datetime import datetime
import ee
import geemap
import sys
import pandas as pd
import geopandas as gp
import matplotlib.pyplot as plt
from datetime import datetime
# from data.geefun import data_info
import geefun as gf
import matplotlib.dates as mdates
from tqdm import tqdm

#%%
ee.Authenticate()
ee.Initialize()

#%%
sys.path.insert(0,'src')
import config

sel_var = 'chirps' # gpm, ndvi

#=========================
# Processing NDVI, Precipitation data
#=========================
# %%
# Note:
# The code was not working when the selected domain is very small such as koreatown in LA. 
# When size is increased it works.

# Read focus region
# gdf = gp.read_file(config.shapefile_dir / "kw/Watershed/Watershedboundary.shp")
# gdf = gp.read_file(config.shapefile_dir / "LosAngeles_ktown.shp")
gdf = gp.read_file(config.shapefile_dir / "Khulna.shp")

gdf = gdf.to_crs('epsg:4326')
gdf.boundary.plot()

area_shp = gdf['geometry'] 
reg = gf.gp_to_ee_poly(area_shp)


# losangeles = ee.Geometry.Polygon([[[-118.2986075736749, 34.0656551967962],
#           [-118.2986075736749, 34.06394873592316],
#           [-118.29611848370908, 34.06394873592316],
#           [-118.29611848370908, 34.0656551967962]]])

# reg = losangeles

#%%
startDate = '2021-01-01'
endDate = '2022-10-31'
geeinfo = gf.data_info()

modisNDVI = geeinfo['modis_ndvi'][0].select(geeinfo['modis_ndvi'][1]).filterDate(startDate, endDate)

if sel_var == 'gpm':
    precip_ee = geeinfo['gpm'][0].select(geeinfo['gpm'][1]).filterDate(startDate, endDate)
if sel_var == 'chirps':
    precip_ee = geeinfo['chirps'][0].select(geeinfo['gpm'][1]).filterDate(startDate, endDate)

if sel_var == 'ndvi':
    eedata = modisNDVI
    eescale = geeinfo['modis_ndvi'][3]
if (sel_var == 'gpm' or sel_var == 'chirps'):
    eedata = precip_ee
    eescale = geeinfo[sel_var][3]

#%%
def get_reg_ee(n):
    date = ee.Date(startDate).advance(n,'month')
    m = date.get("month")
    y = date.get("year")
    dic = ee.Dictionary({
        'Date':date.format('yyyy-MM')
    })
    
    tmpval = (eedata.filter(ee.Filter.calendarRange(y, y, 'year'))
                .filter(ee.Filter.calendarRange(m, m, 'month'))
                .mean()
                .reduceRegion(
                    reducer = ee.Reducer.mean(),
                    geometry = reg,
                    scale = eescale))
    return dic.combine(tmpval)

#%%
#Total number of months
month_count = gf.diff_month(datetime.strptime(endDate, '%Y-%m-%d') ,datetime.strptime(startDate, '%Y-%m-%d'))

if sel_var == 'ndvi':
    ndvi_yrmo = ee.List.sequence(0, month_count).map(get_reg_ee)
    ndvi_df= pd.DataFrame(ndvi_yrmo.getInfo())
    ndvi_df.tail(4)
    gf.get_ts_plot(ndvi_df,label_n='NDVI')

if (sel_var == 'gpm' or sel_var == 'chirps'):
    gpm_yrmo = ee.List.sequence(0, month_count).map(get_reg_ee)
    gpm_df= pd.DataFrame(gpm_yrmo.getInfo())
    gpm_df.tail(4)
    gf.get_ts_plot(gpm_df,label_n = 'Precipitation')


# %%
# Monthly sum relatively slow code
#gpm_mon_tot = gf.get_monthly_mean(geeinfo['gpm'], startDate, endDate, reg)
# %%
#=========================
# Processing crop data
#=========================
# %%
cdl = geeinfo['cdl']
nlcd = geeinfo['nlcd']

crop_df = pd.DataFrame(range(1,254))
crop_df.columns = ['Crop_id']

# reading cdl crop names and corresponding ids
crp_leg = pd.read_csv(config.data_path / 'cdl_legend.csv')
#%%
# join name with crop_df dataframe
crop_df = crop_df.merge(crp_leg,how='outer',left_on=['Crop_id'],right_on=['Crop_id'])

for yr in tqdm(range(2008,2022)):
    #print('Processing year: ' + str(yr))
    #extracting land use types as numbers from gee
    cdl_dict = gf.cdl_pr(yr,reg)
    
    #list the unique land use types
    crop_all = gf.unique(cdl_dict['cropland'])
    
    # counting grid numbers with specific land uses and calculating area in m^2. 30 m cell size
    df_crop = gf.crop_all_area(crop_all,cdl_dict,s_grid=30,attrb = 'cropland')
    df_crop2 = df_crop.T
    df_crop2.columns = ['Crop_id', str('Area_'+ str(yr))]
    df_crop2 = df_crop2.sort_values('Crop_id')

    # join to common dataframe
    crop_df = crop_df.merge(df_crop2,how='outer',left_on=['Crop_id'],right_on=['Crop_id'])
    
# %%
crop_df.head(2)
# %% 
yr_ser = range(2008,2022)
crop_sel = [0,4,35,53,60,68,71,74,75,175,203,211,]
for i in crop_sel:
    val = crop_df.iloc[i,3:]
    c_type = crop_df.iloc[i,1]
    df_pl = pd.DataFrame({'Year': yr_ser, 'Area_sqkm' :list(val)})

    df_pl.plot(x = 'Year', y = 'Area_sqkm', label = c_type,ylabel = 'Area [sq. km]')
# %%


# You can simply copy and paste your GEE JavaScripts into a code block wrapped with trip quotes and pass it to a variable.

# For example, you can grap GEE JavaScripts from GEE Documentation.

js_snippet = """
var image = ee.ImageCollection("NASA/GPM_L3/IMERG_V06")
var data=image.filterDate('2022-11-1','2022-11-8')
print(data)

var ppt=data.map(function(data){
  return data.clip(AOI)
})
var precipitation=ppt.select('precipitationCal')
print(precipitation,'precipitation')
"""
# %%
geemap.js_snippet_to_py(
    js_snippet, add_new_cell=True, import_ee=True, import_geemap=True, show_map=True
)
# %%
