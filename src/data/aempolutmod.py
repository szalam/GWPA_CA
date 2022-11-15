import sys
from pickle import TRUE
import ppfun as dp
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

sys.path.insert(0,'src')
import config


class aempol():

    def __init__(self,cmax_c,c_threshold,gwpa,aemlyrlim, file_aem_interpolated, resistivity_threshold, coord = None, c_above_thres = 1):

        self.cmax_c = cmax_c
        self.threshold = c_threshold
        self.gwpa = gwpa
        self.lyrlim = aemlyrlim
        self.file_aem_interpolated = file_aem_interpolated
        self.aem_resist_threshold = resistivity_threshold
        self.coord = coord
        self.c_above_thres = c_above_thres


    def get_aem_values(self):
        aem_interp = np.load(self.file_aem_interpolated / f'mean_resistivity_upto_layer_{self.lyrlim}.npy')
        aem_interp_X = np.load(self.file_aem_interpolated / 'X_for_resistivity_data.npy')
        aem_interp_Y = np.load(self.file_aem_interpolated / 'Y_for_resistivity_data.npy')

        arr = aem_interp
        # shp = arr.shape
        r = aem_interp_X
        c = aem_interp_Y
        df_res = pd.DataFrame(np.c_[r.ravel(), c.ravel(), arr.ravel()], \
                                        columns=((['x','y','Resistivity'])))
        
        # converting aem data to geodataframe
        gdfres = gpd.GeoDataFrame(
            df_res, geometry=gpd.points_from_xy(df_res.x, df_res.y))
        
        # Convert aem data coordinate to lat lon
        gdfres = gdfres.set_crs(epsg='32611')
        gdfres.to_crs(epsg='4326', inplace=True)

        self.gdfres = gdfres

        # return self.gdfres

    def get_polutdata_inside_aemdomain(self):
        """
        Clip polution data inside the domain of AEM survey
        """

        self.get_aem_values()
        # converting point resistivity data to polygon mask
        gdftmp = self.gdfres.copy()
        gdftmp=gdftmp.dropna(subset=['Resistivity'])

        gdftmp['id_1'] = 1
        gdfmask = gdftmp.dissolve(by="id_1")
        gdfmask["geometry"] = gdfmask["geometry"].convex_hull

        # clipping wq measurement data with the resistivity area mask
        cmax_c_clip = gpd.clip(self.cmax_c, gdfmask)
        if self.c_above_thres == 1:
            cmax_abovthres = cmax_c_clip[cmax_c_clip.VALUE >= self.threshold]
        if self.c_above_thres == 0:
            cmax_abovthres = cmax_c_clip[cmax_c_clip.VALUE < self.threshold]

        self.cmax_c_abovthres = cmax_abovthres

        # return self.cmax_c_abovthres 

    def get_aem_at_wqnodes(self):
        
        """
        Get aem data at water quality nodes
        """

        self.get_polutdata_inside_aemdomain()

        # spatial join of wq data and aem data
        aem_cmax = gpd.sjoin_nearest(self.cmax_c_abovthres, self.gdfres)
        aem_cmax=aem_cmax.dropna(subset=['Resistivity'])
        
        # after spatial join there could be one colum with index_right title. Removing this if exists
        aem_cmax_gwpa = aem_cmax.drop(['index_right'], axis=1, errors='ignore')
        
        # spatial join of aem data and gwpa
        aem_cmax_gwpa = gpd.sjoin(aem_cmax_gwpa, self.gwpa, how='left')
        
        # getting ids for rows overlap gwpa
        selected_rows = aem_cmax_gwpa['GWPAType'].isnull()

        # separting aem data insude and outside gwpa
        aem_cmax_gwpa_out = aem_cmax_gwpa[selected_rows]
        aem_cmax_gwpa_in = aem_cmax_gwpa[~selected_rows]

        self.aem_cmax_gwpa_out = aem_cmax_gwpa_out
        self.aem_cmax_gwpa_in = aem_cmax_gwpa_in

        # return self.aem_cmax_gwpa_in, self.aem_cmax_gwpa_out


    def get_scatter_aem_polut(self, yaxis_lim = None, gwpaflag = None):
        """
        Get scatter plot of aem values and pollutant concentration

        Arg:
        gwpaflag: None = outside gwpa, 1 = insude gwpa, All = both
        """

        self.get_aem_at_wqnodes()

        fig, ax = plt.subplots(figsize=(7, 7))
        if gwpaflag is None:
            ax.scatter(self.aem_cmax_gwpa_out.Resistivity,self.aem_cmax_gwpa_out.VALUE,color = 'red', label = 'Outside GWPA', s = 1)
        if gwpaflag == 1:
            ax.scatter(self.aem_cmax_gwpa_in.Resistivity,self.aem_cmax_gwpa_in.VALUE,color = 'blue', label = 'Inside GWPA', s = .8)
        if gwpaflag == 'All':
            ax.scatter(self.aem_cmax_gwpa_in.Resistivity,self.aem_cmax_gwpa_in.VALUE,color = 'red', label = 'Inside GWPA', s = 1)
            ax.scatter(self.aem_cmax_gwpa_out.Resistivity,self.aem_cmax_gwpa_out.VALUE,color = 'blue', label = 'Outside GWPA', s = .8)
        
        plt.xlim([0, 100])
        if yaxis_lim is None:
            plt.ylim([0,self.aem_cmax_gwpa_out.VALUE.max()])
        else:
            plt.ylim([0,yaxis_lim])

        plt.legend(loc="upper left")
        plt.xlabel('Electrical Reistivity', fontsize=20)
        plt.ylabel('Nitrate concentration [mg/l]', fontsize=20)
        plt.legend(fontsize=17) 
        plt.tick_params(axis='both', which='major', labelsize=17)
        plt.show()

    def get_aem_hist_inout_gwpa(self):

        self.get_aem_at_wqnodes()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.hist(self.aem_cmax_gwpa_in.Resistivity, density=True, bins=50, label = 'Inside GWPA',alpha=0.5)  # density=False would make counts
        ax.hist(self.aem_cmax_gwpa_out.Resistivity, density=True, bins=50, label = 'Outside GWPA',alpha=0.5)  # density=False would make counts
        plt.ylabel('Probability', fontsize = 20)
        plt.xlabel('Electrical Reistivity', fontsize=20)
        plt.xlim([0, 250])
        plt.legend(loc="upper right")
        plt.legend(fontsize=17) 
        plt.tick_params(axis='both', which='major', labelsize=17)
        plt.show()

    def get_resistivity_plot_abovthresh(self):
        gdfres_thres = self.gdfres[self.gdfres['Resistivity']>self.aem_resist_threshold]

        fig, ax = plt.subplots(1,1,figsize = (10,10))
        gdfres_thres.plot(ax = ax, label = 'Vulnerable zone')
        self.gwpa.plot(ax = ax, label = 'GWPA',facecolor = 'none', edgecolor = 'red', lw = .5, zorder=3, alpha = .8)
        plt.legend(loc="lower left")
        plt.legend(fontsize=17) 
        plt.tick_params(axis='both', which='major', labelsize=14)