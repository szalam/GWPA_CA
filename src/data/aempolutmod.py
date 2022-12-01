import sys
from pickle import TRUE
import ppfun as dp
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import warnings

sys.path.insert(0,'src')
import config


class aempol():

    def __init__(self,cmax_c,c_threshold,gwpa,aemlyrlim,file_loc_interpolated, file_aem_interpolated, resistivity_threshold, coord = None, c_above_thres = 1,aemsrc = "DWR",aemregion = 5, aem_value_type = 'conductivity', gwd_gdf = None, sagbi = None,rad_mile = 2):

        """
        Arg:
        rad_mile: radius of well buffer. Default 2
        """

        self.cmax_c = cmax_c
        self.threshold = c_threshold
        self.gwpa = gwpa
        self.lyrlim = aemlyrlim
        self.file_loc_interpolated = file_loc_interpolated
        self.file_aem_interpolated = file_aem_interpolated
        self.aem_resist_threshold = resistivity_threshold
        self.coord = coord
        self.c_above_thres = c_above_thres
        self.aemsrc = aemsrc
        self.aemregion = aemregion
        self.aem_value_type = aem_value_type
        self.gwd_gdf = gwd_gdf
        self.sagbi = sagbi
        self.rad_mile = rad_mile


    def get_aem_values(self):
        """
        Get AEM resistivity values as geodataframe. Though the column name for AEM value is referred as
        Resistivity thoughout the model, it will indicate conductivity if the imported data is conductivity. 
        """
        aem_interp = np.load(self.file_loc_interpolated / f'{self.file_aem_interpolated}')
        if self.aemsrc == 'DWR':
            aem_interp_X = np.load(self.file_loc_interpolated / f'X_region{self.aemregion}.npy')
            aem_interp_Y = np.load(self.file_loc_interpolated / f'Y_region{self.aemregion}.npy')
        if self.aemsrc == 'ENVGP':
            aem_interp_X = np.load(self.file_loc_interpolated / f'X.npy')
            aem_interp_Y = np.load(self.file_loc_interpolated / f'Y.npy')

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
        if self.aemsrc == 'DWR':
            gdfres = gdfres.set_crs(epsg='3310')
        if self.aemsrc == 'ENVGP':
            gdfres = gdfres.set_crs(epsg='32611')

        gdfres.to_crs(epsg='4326', inplace=True)

        self.gdfres = gdfres

        return self.gdfres

    def get_aem_mask(self):
        """
        Create mask around AEM dataset
        """
        self.get_aem_values()

        # converting point resistivity data to polygon mask
        gdftmp = self.gdfres.copy()
        gdftmp=gdftmp.dropna(subset=['Resistivity'])

        gdftmp['id_1'] = 1
        gdfmask = gdftmp.dissolve(by="id_1")
        gdfmask["geometry"] = gdfmask["geometry"].convex_hull

        self.gdfmask = gdfmask

        return self.gdfmask
        

    def get_polutdata_inside_aemdomain(self):
        """
        Clip polution data inside the domain of AEM survey
        """

        self.get_aem_mask()

        # clipping wq measurement data with the resistivity area mask
        cmax_c_clip = gpd.clip(self.cmax_c, self.gdfmask)
        if self.c_above_thres == 1:
            cmax_abovthres = cmax_c_clip[cmax_c_clip.VALUE >= self.threshold]
        if self.c_above_thres == 0:
            cmax_abovthres = cmax_c_clip[cmax_c_clip.VALUE < self.threshold]

        self.cmax_c_abovthres = cmax_abovthres
        
        return self.cmax_c_abovthres 

    def get_aem_at_wqnodes(self):
        
        """
        Get aem data at water quality nodes. 
        """

        # separating pollution data inside aem domain
        self.get_polutdata_inside_aemdomain()

        # spatial join of wq data and aem data
        aem_cmax = gpd.sjoin_nearest(self.cmax_c_abovthres, self.gdfres)
        aem_cmax=aem_cmax.dropna(subset=['Resistivity'])
        
        # after spatial join there could be one colum with index_right title. Removing this if exists
        aem_cmax_gwpa = aem_cmax.drop(['index_right'], axis=1, errors='ignore')
        
        # spatial join of aem data and gwpa
        aem_cmax_gwpa = gpd.sjoin(aem_cmax_gwpa, self.gwpa, how='left')
        
        self.aem_cmax_gwpa = aem_cmax_gwpa

        return self.aem_cmax_gwpa

    def get_aem_at_wqnodes_inout_gwpa(self):
        """
        Separate AEM data inside and outside groundwater protection area (GWPA)
        """
        aem_cmax_gwpa = self.get_aem_at_wqnodes()

        # getting ids for rows overlap gwpa
        selected_rows = aem_cmax_gwpa['GWPAType'].isnull()

        # separting aem data insude and outside gwpa
        aem_cmax_gwpa_out = aem_cmax_gwpa[selected_rows]
        aem_cmax_gwpa_in = aem_cmax_gwpa[~selected_rows]

        self.aem_cmax_gwpa_out = aem_cmax_gwpa_out
        self.aem_cmax_gwpa_in = aem_cmax_gwpa_in

        return self.aem_cmax_gwpa_in, self.aem_cmax_gwpa_out
        
    def get_gwdepth_at_wqnodes(self):
        
        """
        Get groundwater depth data at water quality nodes
        """

        self.get_polutdata_inside_aemdomain()

        # spatial join of wq data and aem data
        aem_cmax = gpd.sjoin_nearest(self.cmax_c_abovthres, self.gdfres)
        aem_cmax=aem_cmax.dropna(subset=['Resistivity'])
        
        # after spatial join there could be one colum with index_right title. Removing this if exists
        aem_cmax_gwpa = aem_cmax.drop(['index_right'], axis=1, errors='ignore')
        
        # # spatial join of aem data and gwpa
        # aem_cmax_gwpa = gpd.sjoin(aem_cmax_gwpa, self.gwpa, how='left')
        
        # aem_cmax_gwpa = aem_cmax.drop(['index_right'], axis=1, errors='ignore')

        # spatial join of wq data and gw depth data
        gwd_cmax = gpd.sjoin_nearest(aem_cmax_gwpa, self.gwd_gdf)
        
        # after spatial join there could be one colum with index_right title. Removing this if exists
        gwd_cmax_gwpa = gwd_cmax.drop(['index_right'], axis=1, errors='ignore')
        
        self.gwd_cmax_gwpa = gwd_cmax_gwpa

        return self.gwd_cmax_gwpa


    def get_sagbi_at_wqnodes(self):
        
        """
        Get sagbi data at water quality measurement locations.
        """

        self.get_polutdata_inside_aemdomain()

        # spatial join of wq data and aem data
        aem_cmax = gpd.sjoin_nearest(self.cmax_c_abovthres, self.gdfres)
        aem_cmax=aem_cmax.dropna(subset=['Resistivity'])
        
        # after spatial join there could be one colum with index_right title. Removing this if exists
        aem_cmax2 = aem_cmax.drop(['index_right'], axis=1, errors='ignore')
        
        # spatial join of wq data and gw depth data
        gwd_cmax = gpd.sjoin_nearest(aem_cmax2, self.sagbi)
        
        # after spatial join there could be one colum with index_right title. Removing this if exists
        gwd_cmax2 = gwd_cmax.drop(['index_right'], axis=1, errors='ignore')
        
        self.sagbi_cmax = gwd_cmax2

        return self.sagbi_cmax



    def get_scatter_aem_polut(self, datareg2 = None, yaxis_lim = None, xaxis_lim = None, gwpaflag = None, YlabelC = 'Nitrate',unitylabel = 'mg/l'):
        """
        Get scatter plot of aem values and pollutant concentration

        Arg:
        
        yaxis_lim; xaxis_lim: None = uses max value, user input= uses max value
        gwpaflag: None = outside gwpa, 1 = insude gwpa, both = both, All: plot all
        YlabelC: yaxis label
        unitylabel: y-axis unit
        """

        self.get_aem_at_wqnodes()
        self.get_aem_at_wqnodes_inout_gwpa()

        if datareg2 is not None:
            reg2_in, reg2_out = datareg2.get_aem_at_wqnodes_inout_gwpa()
            dfall_in = pd.concat([self.aem_cmax_gwpa_in, reg2_in])
            dfall_out = pd.concat([self.aem_cmax_gwpa_out, reg2_out])
            dfall_all = datareg2.get_aem_at_wqnodes()

        if datareg2 is None:
            dfall_in = self.aem_cmax_gwpa_in
            dfall_out = self.aem_cmax_gwpa_out
            dfall_all = self.get_aem_at_wqnodes()

        fig, ax = plt.subplots(figsize=(7, 7))
        if gwpaflag is None:
            ax.scatter(dfall_out.Resistivity,dfall_out.VALUE,color = 'red', label = 'Outside GWPA', s = 1)
        if gwpaflag == 1:
            ax.scatter(dfall_in.Resistivity,dfall_in.VALUE,color = 'blue', label = 'Inside GWPA', s = .8)
        if gwpaflag == 'both':
            ax.scatter(dfall_in.Resistivity,dfall_in.VALUE,color = 'red', label = 'Inside GWPA', s = 1)
            ax.scatter(dfall_out.Resistivity,dfall_out.VALUE,color = 'blue', label = 'Outside GWPA', s = .8)
        if gwpaflag == 'All':
            ax.scatter(dfall_all.Resistivity,dfall_all.VALUE,color = 'red', s = 1)
        
        if xaxis_lim is not None:
            plt.xlim([0, xaxis_lim])

        if yaxis_lim is None:
            plt.ylim([0,dfall_all.VALUE.max()])
        else:
            plt.ylim([0,yaxis_lim])

        if gwpaflag != 'All':
            plt.legend(loc="upper left")
            plt.legend(fontsize=17) 

        if self.aem_value_type == 'resistivity':
            plt.xlabel('Electrical Reistivity', fontsize=20)
        if self.aem_value_type == 'conductivity':
            plt.xlabel('Conductivity', fontsize=20)

        plt.ylabel(f'{YlabelC} concentration [{unitylabel}]', fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=17)
        plt.show()


    # def get_scatter_aem_polut_tworegions(self, datareg2, yaxis_lim = None, xaxis_lim = None, gwpaflag = None, YlabelC = 'Nitrate',unitylabel = 'mg/l'):
    #     """
    #     Get scatter plot of aem values and pollutant concentration

    #     Arg:
    #     gwpaflag: None = outside gwpa, 1 = insude gwpa, All = both
    #     """

    #     self.get_aem_at_wqnodes()
    #     reg2_in, reg2_out = datareg2.get_aem_at_wqnodes()

    #     dfall_in = pd.concat([self.aem_cmax_gwpa_in, reg2_in])
    #     dfall_out = pd.concat([self.aem_cmax_gwpa_out, reg2_out])
    #     fig, ax = plt.subplots(figsize=(7, 7))
    #     if gwpaflag is None:
    #         ax.scatter(dfall_out.Resistivity,dfall_out.VALUE,color = 'red', label = 'Outside GWPA', s = 1)
    #     if gwpaflag == 1:
    #         ax.scatter(dfall_in.Resistivity,dfall_in.VALUE,color = 'blue', label = 'Inside GWPA', s = .8)
    #     if gwpaflag == 'both':
    #         ax.scatter(dfall_in.Resistivity,dfall_in.VALUE,color = 'red', label = 'Inside GWPA', s = 1)
    #         ax.scatter(dfall_out.Resistivity,dfall_out.VALUE,color = 'blue', label = 'Outside GWPA', s = .8)
    #     if gwpaflag == 'All':
    #         ax.scatter(dfall_in.Resistivity,dfall_in.VALUE,color = 'red', s = 1)
    #         ax.scatter(dfall_out.Resistivity,dfall_out.VALUE,color = 'red', s = .8)
        
    #     if xaxis_lim is not None:
    #         plt.xlim([0, xaxis_lim])

    #     if yaxis_lim is None:
    #         plt.ylim([0, dfall_out.VALUE.max()])
    #     else:
    #         plt.ylim([0,yaxis_lim])

    #     if gwpaflag != 'All':
    #         plt.legend(loc="upper left")
    #         plt.legend(fontsize=17) 

    #     if self.aem_value_type == 'resistivity':
    #         plt.xlabel('Electrical Reistivity', fontsize=20)
    #     if self.aem_value_type == 'conductivity':
    #         plt.xlabel('Conductivity', fontsize=20)

    #     plt.ylabel(f'{YlabelC} concentration [{unitylabel}]', fontsize=20)
    #     plt.tick_params(axis='both', which='major', labelsize=17)
    #     plt.show()


    def get_scatter_aem_polut_w_wellproperty(self, yaxis_lim = None, wellproperty = None, p_size = .8, YlabelC = 'Nitrate'):
        """
        Get scatter plot of aem values and pollutant concentration and color with well property

        Arg:
        yaxis_lim    = y-axis limit of the plot. If None, it uses the max 
        wellproperty = well property column name interested to be compared 
        p_size       = size of point.
        YlabelC      = y-axis label
        """

        self.get_aem_at_wqnodes()

        fig, ax = plt.subplots(figsize=(7, 7))
        
        sc = ax.scatter(self.aem_cmax_gwpa.Resistivity,self.aem_cmax_gwpa.VALUE,c = self.aem_cmax_gwpa[wellproperty], s = p_size, cmap='viridis')
        plt.colorbar(sc)

        plt.xlim([0, 100])
        if yaxis_lim is None:
            plt.ylim([0,self.aem_cmax_gwpa_out.VALUE.max()])
        else:
            plt.ylim([0,yaxis_lim])

        # plt.legend(loc="upper left")
        if self.aem_value_type == 'resistivity':
            plt.xlabel('Electrical Reistivity', fontsize=20)
        if self.aem_value_type == 'conductivity':
            plt.xlabel('Conductivity', fontsize=20)

        plt.ylabel(f'{YlabelC} concentration [mg/l]', fontsize=20)
        # plt.legend(fontsize=17) 
        plt.tick_params(axis='both', which='major', labelsize=17)
        plt.show()

        
    def get_scatter_aem_polut_w_gwdepth(self, yaxis_lim = None, xaxis_lim = None, p_size = .8, YlabelC = 'Nitrate'):
        """
        Get scatter plot of aem values and pollutant concentration, with color of groundwater depth

        Arg:
        yaxis_lim = y-axis limit of the plot. If None, it uses the max 
        xaxis_lim = x-axis limit of the plot. If None, it uses the max 
        p_size    = size of point.
        YlabelC   = y-axis label
        """

        # self.get_aem_at_wqnodes()
        self.get_gwdepth_at_wqnodes()

        fig, ax = plt.subplots(figsize=(7, 7))
        
        sc = ax.scatter(self.gwd_cmax_gwpa.Resistivity,self.gwd_cmax_gwpa.VALUE,c = self.gwd_cmax_gwpa.gwdep, s = p_size, cmap='viridis')
        plt.colorbar(sc)

        if xaxis_lim is not None:
            plt.xlim([0, xaxis_lim])

        if yaxis_lim is None:
            plt.ylim([0,self.gwd_cmax_gwpa.VALUE.max()])
        else:
            plt.ylim([0,yaxis_lim])

        # plt.legend(loc="upper left")
        if self.aem_value_type == 'resistivity':
            plt.xlabel('Electrical Reistivity', fontsize=20)
        if self.aem_value_type == 'conductivity':
            plt.xlabel('Conductivity', fontsize=20)

        plt.ylabel(f'{YlabelC} concentration [mg/l]', fontsize=20)
        # plt.legend(fontsize=17) 
        plt.tick_params(axis='both', which='major', labelsize=17)
        plt.show()


    def create_well_buffer(self):
        """
        Create well buffer
        
        Arg:
        rad_mile: well buffer radius in miles
        """

        self.get_sagbi_at_wqnodes()
        wqnodes = self.sagbi_cmax
        wqnodes['serial'] = wqnodes.index

        #create buffer
        wqnodes_2mi = wqnodes.buffer(self.rad_mile/69) # 1 degree = 69 miles.

        #convert geoseries to gedataframe
        wqnodes_2m_gpd= gpd.GeoDataFrame(geometry=gpd.GeoSeries(wqnodes_2mi))
        wqnodes_2m_gpd['serial'] = wqnodes_2m_gpd.index
        #there will be two geometry column in next line. Removing this here as not needed later
        wqnodes_tmp = wqnodes.drop(['geometry'], axis=1, errors='ignore') 
        wqnodes_2m_gpd = wqnodes_2m_gpd.merge(wqnodes_tmp, on='serial', how='left')
        
        # using wells needed later
        wqnodes_2m_gpd = wqnodes_2m_gpd[['WELL ID','VALUE','geometry']]

        self.wqnodes_2m_gpd = wqnodes_2m_gpd

        return self.wqnodes_2m_gpd

    def get_sagbi_at_wq_buffer(self):

        """
        Get water quality buffers with average sagbi rating. combine wq values and sagbi rating
        """
        self.create_well_buffer()
        g = gpd.overlay(self.sagbi,self.wqnodes_2m_gpd, how='intersection')

        # area weighted sagbi rating calculate
        g['Area_calc'] =g.apply(lambda row: row.geometry.area,axis=1)
        g['area_sagbi'] = g['Area_calc'] * g['sagbi']
    
        dff = g.groupby(["WELL ID"]).Area_calc.sum().reset_index()
        dff2 = g.groupby(["WELL ID"]).area_sagbi.sum().reset_index()
        dff2['area_wt_sagbi'] = dff2['area_sagbi']/ dff['Area_calc']

        dff2 = dff2.drop(['area_sagbi'], axis=1, errors='ignore')
        
        # merge wq data 
        self.wqbuff_wt_sagbi = g.merge(dff2, on='WELL ID', how='left')

        return self.wqbuff_wt_sagbi

    def get_aem_mean_in_well_buffer(self, datareg2 = None):
        """
        Get aem mean in well buffer
        Arg:
        regbound: aem data boundary such as 'cv' for central valley
        datareg2: region 2 data
        """
        
        if datareg2 is not None:
            aemdata = self.get_combined_aem_two_regions(datareg2)
        
        aem_wq_buff = gpd.overlay(aemdata,self.wqnodes_2m_gpd, how='intersection')
        aem_wq_buff_aemmean = aem_wq_buff.groupby(["WELL ID"]).Resistivity.mean().reset_index()
    
        aem_wq_buff_aemmean = aem_wq_buff_aemmean.merge(self.wqnodes_2m_gpd, on='WELL ID', how='left')
        
        self.aem_wq_buff_aemmean = aem_wq_buff_aemmean
        return self.aem_wq_buff_aemmean

    def get_scatter_aem_polut_w_sagbi(self,datareg2 = None,sagbi_rating_min = None, 
                                    sagbi_rating_max = None, sagbi_var = 'rat_grp', 
                                    yaxis_lim = None, xaxis_lim = None, p_size = .8, 
                                    YlabelC = 'Nitrate'):
        """
        Get scatter plot of aem values and pollutant concentration, with color of groundwater depth

        Arg:
        datareg2           = data for another region to be plotted
        sagbi_rating_min    = plotting min range
        sagbi_rating_max    = plotting max range
        sagbi_var           = sagbi variable to plot
        yaxis_lim           = y-axis limit for plot
        xaxis_lim           = x-axis limit for plot
        p_size              = marker size for plot
        YlabelC             = y-axis label for plot
        """

        warnings.filterwarnings("ignore")
        # self.get_aem_at_wqnodes()
        self.get_sagbi_at_wqnodes()

        if datareg2 is not None:
            aemres_sagbi = pd.concat([self.get_sagbi_at_wqnodes(), datareg2.get_sagbi_at_wqnodes()])
        if datareg2 is None:
            aemres_sagbi = self.get_sagbi_at_wqnodes().copy()

        if sagbi_rating_min is not None:
            aemres_sagbi = aemres_sagbi[aemres_sagbi.sagbi>=sagbi_rating_min]
        if sagbi_rating_max is not None:
            aemres_sagbi = aemres_sagbi[aemres_sagbi.sagbi<=sagbi_rating_max]


        fig, ax = plt.subplots(figsize=(7, 7))
        
        if sagbi_var == 'rat_grp':
            sc = ax.scatter(aemres_sagbi.Resistivity,aemres_sagbi.VALUE,c = aemres_sagbi.rat_grp, cmap='viridis')
        if sagbi_var == 'sagbi':
            sc = ax.scatter(aemres_sagbi.Resistivity,aemres_sagbi.VALUE,c = aemres_sagbi.sagbi, s = p_size, cmap='viridis')
        if sagbi_var == 'rt_zn_res_':
            sc = ax.scatter(aemres_sagbi.Resistivity,aemres_sagbi.VALUE,c = aemres_sagbi.rt_zn_res_, s = p_size, cmap='viridis')
        if sagbi_var == 'surf_cond':
            sc = ax.scatter(aemres_sagbi.Resistivity,aemres_sagbi.VALUE,c = aemres_sagbi.surf_cond, s = p_size, cmap='viridis')
        if sagbi_var == 'topo_rest':
            sc = ax.scatter(aemres_sagbi.Resistivity,aemres_sagbi.VALUE,c = aemres_sagbi.topo_rest, s = p_size, cmap='viridis')
        if sagbi_var == 'restrictns':
            sc = ax.scatter(aemres_sagbi.Resistivity,aemres_sagbi.VALUE,c = aemres_sagbi.restrictns,  cmap='viridis')

        plt.colorbar(sc)

        if xaxis_lim is not None:
            plt.xlim([0, xaxis_lim])

        if yaxis_lim is None:
            plt.ylim([0,self.gwd_cmax_gwpa.VALUE.max()])
        else:
            plt.ylim([0,yaxis_lim])

        # plt.legend(loc="upper left")
        if self.aem_value_type == 'resistivity':
            plt.xlabel('Electrical Reistivity', fontsize=20)
        if self.aem_value_type == 'conductivity':
            plt.xlabel('Conductivity', fontsize=20)

        plt.ylabel(f'{YlabelC} concentration [mg/l]', fontsize=20)
        # plt.legend(fontsize=17) 
        plt.tick_params(axis='both', which='major', labelsize=17)
        plt.show()


    def get_scatter_sagbi_aem_in_buffer(self, datareg2 = None):
        """
        Scatter plot of sagbi and aem resisitivity/conductivity data in buffer zone
        """
        self.get_sagbi_at_wq_buffer()

        if datareg2 is not None:
            sagbi_regs = pd.concat([self.get_sagbi_at_wq_buffer(), datareg2.get_sagbi_at_wq_buffer()])
        if datareg2 is None:
            sagbi_regs = self.get_sagbi_at_wq_buffer()

        self.get_aem_mean_in_well_buffer(datareg2 = datareg2)

        sagbi_aem = sagbi_regs.merge(self.aem_wq_buff_aemmean, on='WELL ID', how='left')

        fig, ax = plt.subplots(figsize=(7, 7))
        plt.scatter(sagbi_aem['area_wt_sagbi'], sagbi_aem['Resistivity'],
                    color = 'red', label = 'Outside GWPA', s = 1)
                    
        plt.ylim([0,1])
        plt.xlabel('SAGBI Rating', fontsize=20)

        # plt.legend(loc="upper left")
        if self.aem_value_type == 'resistivity':
            plt.ylabel('Electrical Reistivity', fontsize=20)
        if self.aem_value_type == 'conductivity':
            plt.ylabel('Conductivity', fontsize=20)

        plt.tick_params(axis='both', which='major', labelsize=17)
        plt.show()

    def get_scatter_sagbi_polut_in_buffer(self,ylim_sel = None):
        """
        Scatter plot of sagbi and polution data in buffer zone
        """
        self.get_sagbi_at_wq_buffer()

        fig, ax = plt.subplots(figsize=(7, 7))
        plt.scatter(self.wqbuff_wt_sagbi['area_wt_sagbi'],self.wqbuff_wt_sagbi['VALUE'],
                    color = 'red', label = 'Outside GWPA', s = 1)
        if ylim_sel is not None:
            plt.ylim([0,ylim_sel])
            
        plt.xlabel('SAGBI Rating', fontsize=20)
        plt.ylabel('Nitrate concentration [mg/l]', fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=17)
        plt.show()


    def get_scatter_aem_vs_sagbi(self,datareg2 = None, sagbi_var = 'rat_grp', 
                                    sagbi_rating_min = None, sagbi_rating_max = None, 
                                    yaxis_lim = None, xaxis_lim = None, p_size = .8, 
                                    YlabelC = 'Nitrate'):
        """
        Get scatter plot of aem values and pollutant concentration, with color of groundwater depth

        Arg:
        gwpaflag: None = outside gwpa, 1 = insude gwpa, All = both
        """

        warnings.filterwarnings("ignore")

        self.get_sagbi_at_wqnodes()

        if datareg2 is not None:
            datareg2.get_sagbi_at_wqnodes()
            aemres_sagbi = pd.concat([self.sagbi_cmax, datareg2.sagbi_cmax])
        if datareg2 is None:
            aemres_sagbi = self.self.sagbi_cmax.copy()

        if sagbi_rating_min is not None:
            aemres_sagbi = aemres_sagbi[aemres_sagbi.sagbi>=sagbi_rating_min]
        if sagbi_rating_max is not None:
            aemres_sagbi = aemres_sagbi[aemres_sagbi.sagbi<=sagbi_rating_max]

        fig, ax = plt.subplots(figsize=(7, 7))
        
        if sagbi_var == 'rat_grp':
            sc = ax.scatter(aemres_sagbi.Resistivity,aemres_sagbi.rat_grp, s = p_size, cmap='viridis')
        if sagbi_var == 'sagbi':
            sc = ax.scatter(aemres_sagbi.Resistivity,aemres_sagbi.sagbi, s = p_size, cmap='viridis')
        if sagbi_var == 'rt_zn_res_':
            sc = ax.scatter(aemres_sagbi.Resistivity,aemres_sagbi.rt_zn_res_, s = p_size, cmap='viridis')
        if sagbi_var == 'surf_cond':
            sc = ax.scatter(aemres_sagbi.Resistivity,aemres_sagbi.surf_cond, s = p_size, cmap='viridis')
        if sagbi_var == 'topo_rest':
            sc = ax.scatter(aemres_sagbi.Resistivity,aemres_sagbi.topo_rest, s = p_size, cmap='viridis')
        if sagbi_var == 'restrictns':
            sc = ax.scatter(aemres_sagbi.Resistivity,aemres_sagbi.restrictns, s = p_size, cmap='viridis')

        # plt.colorbar(sc)

        if xaxis_lim is not None:
            plt.xlim([0, xaxis_lim])

        if yaxis_lim is None:
            plt.ylim([0,self.aemres_sagbi.VALUE.max()])
        else:
            plt.ylim([0,yaxis_lim])

        # plt.legend(loc="upper left")
        if self.aem_value_type == 'resistivity':
            plt.xlabel('Electrical Reistivity', fontsize=20)
        if self.aem_value_type == 'conductivity':
            plt.xlabel('Conductivity', fontsize=20)

        plt.ylabel(f'{YlabelC} Sagbi values', fontsize=20)
        # plt.legend(fontsize=17) 
        plt.tick_params(axis='both', which='major', labelsize=17)
        plt.show()



    def get_scatter_polut_vs_sagbi(self,datareg2 = None, sagbi_var = 'rat_grp', yaxis_lim = None, xaxis_lim = None, p_size = .8, YlabelC = 'Nitrate'):
        """
        Get scatter plot of aem values and pollutant concentration, with color of groundwater depth

        Arg:
        gwpaflag: None = outside gwpa, 1 = insude gwpa, All = both
        """

        warnings.filterwarnings("ignore")
        # self.get_aem_at_wqnodes()
        self.get_sagbi_at_wqnodes()
        if datareg2 is not None:
            aemres_sagbi = pd.concat([self.get_sagbi_at_wqnodes(), datareg2.get_sagbi_at_wqnodes()])
        if datareg2 is None:
            aemres_sagbi = self.get_sagbi_at_wqnodes().copy()

        fig, ax = plt.subplots(figsize=(7, 7))
        
        if sagbi_var == 'rat_grp':
            sc = ax.scatter(aemres_sagbi.rat_grp,aemres_sagbi.VALUE, s = p_size, cmap='viridis')
        if sagbi_var == 'sagbi':
            sc = ax.scatter(aemres_sagbi.sagbi,aemres_sagbi.VALUE, s = p_size, cmap='viridis')
        if sagbi_var == 'rt_zn_res_':
            sc = ax.scatter(aemres_sagbi.rt_zn_res_, aemres_sagbi.VALUE, s = p_size, cmap='viridis')
        if sagbi_var == 'surf_cond':
            sc = ax.scatter(aemres_sagbi.surf_cond, aemres_sagbi.VALUE, s = p_size, cmap='viridis')
        if sagbi_var == 'topo_rest':
            sc = ax.scatter(aemres_sagbi.topo_rest, aemres_sagbi.VALUE, s = p_size, cmap='viridis')
        if sagbi_var == 'restrictns':
            sc = ax.scatter(aemres_sagbi.restrictns, aemres_sagbi.VALUE, s = p_size, cmap='viridis')

        # plt.colorbar(sc)

        if xaxis_lim is not None:
            plt.xlim([0, xaxis_lim])

        if yaxis_lim is None:
            plt.ylim([0,self.aemres_sagbi.VALUE.max()])
        else:
            plt.ylim([0,yaxis_lim])

        # plt.legend(loc="upper left")
        if self.aem_value_type == 'resistivity':
            plt.xlabel('Electrical Reistivity', fontsize=20)
        if self.aem_value_type == 'conductivity':
            plt.xlabel('Conductivity', fontsize=20)

        plt.ylabel(f'{YlabelC} Sagbi values', fontsize=20)
        # plt.legend(fontsize=17) 
        plt.tick_params(axis='both', which='major', labelsize=17)
        plt.show()

    def get_aem_hist_inout_gwpa(self,xlim_sel = 100):

        self.get_aem_at_wqnodes()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.hist(self.aem_cmax_gwpa_in.Resistivity, density=True, bins=50, label = 'Inside GWPA',alpha=0.5)  # density=False would make counts
        ax.hist(self.aem_cmax_gwpa_out.Resistivity, density=True, bins=50, label = 'Outside GWPA',alpha=0.5)  # density=False would make counts
        plt.ylabel('Probability', fontsize = 20)
        if self.aem_value_type == 'resistivity':
            plt.xlabel('Electrical Reistivity', fontsize=20)
        if self.aem_value_type == 'conductivity':
            plt.xlabel('Conductivity', fontsize=20)
        plt.xlim([0, xlim_sel])
        plt.legend(loc="upper right")
        plt.legend(fontsize=17) 
        plt.tick_params(axis='both', which='major', labelsize=17)
        plt.show()

    def get_risk_scores(self):

        gdfaem = self.aem_cmax_gwpa_out.copy()
        gdfaem['risk_score'] = 0

        for i in range(0,70,10):
            sel_ids = (gdfaem['Resistivity'] > i) & (gdfaem['Resistivity'] <= i+10)
            # Separate data within the date ranges
            gdfaem['risk_score'].loc[sel_ids] = i

        return gdfaem

    def get_aemzone_plot_abovthresh(self, aem_threshold = None):
        if aem_threshold is None:
            gdfres_thres = self.gdfres[self.gdfres['Resistivity']>self.aem_resist_threshold]
        if aem_threshold is not None:
            gdfres_thres = self.gdfres[self.gdfres['Resistivity']>aem_threshold]

        fig, ax = plt.subplots(1,1,figsize = (10,10))
        gdfres_thres.plot(ax = ax, label = 'Vulnerable zone')
        self.gwpa.plot(ax = ax, label = 'GWPA',facecolor = 'none', edgecolor = 'red', lw = .5, zorder=3, alpha = .8)
        plt.legend(loc="lower left")
        plt.legend(fontsize=17) 
        plt.tick_params(axis='both', which='major', labelsize=14)

    
    def get_aem_riskscore_plot(self,reg,risk_interval = 1):

        self.get_aem_values()
        aemres = self.gdfres.copy()
        aemres=aemres.dropna(subset=['Resistivity'])

        aemres['risk_score'] = 0

        N = 1000
        count_tmp = 0
        for i in np.arange(0.0, round(aemres['Resistivity'].max(),1), risk_interval):
            sel_ids = (aemres['Resistivity'] > i) & (aemres['Resistivity'] <= i+risk_interval)
            # Separate data within the date ranges
            aemres['risk_score'].loc[sel_ids] = N
            N=N-1
            count_tmp =count_tmp+1
        aemres['risk_score'] = aemres['risk_score']  - (1000-N)


            
        fig = plt.figure(figsize=(10, 10))
        reg.plot(label = 'Kaweah',facecolor = 'none', edgecolor = 'black', lw = .5, zorder = 10)
        out = plt.scatter(
            aemres['geometry'].x, aemres['geometry'].y, c=aemres.risk_score, 
            s=10, 
            cmap='turbo',
            zorder = 1
        )
        plt.xlabel("Easting (m)")
        plt.ylabel("Northing (m)")
        cbar = plt.colorbar(out, fraction=0.03)
        cbar.set_label('Risk scores') 
        plt.axis(False)

    def get_aem_resistivity_plot(self,kw):

        self.get_aem_values()
        aemres = self.gdfres.copy()
        aemres=aemres.dropna(subset=['Resistivity'])
        fig = plt.figure(figsize=(10, 10))
        kw.plot(label = 'Kaweah',facecolor = 'none', edgecolor = 'black', lw = .5, zorder = 10)
        out = plt.scatter(
            aemres['geometry'].x, aemres['geometry'].y, c=aemres.Resistivity, 
            s=10, 
            norm=LogNorm(vmin=20, vmax=100), 
            cmap='turbo',
            zorder = 1
        )
        plt.xlabel("Easting (m)")
        plt.ylabel("Northing (m)")
        cbar = plt.colorbar(out, fraction=0.03)
        if self.aem_value_type == 'resistivity':
            cbar.set_label('Resistivity ($\Omega$m)') 
        if self.aem_value_type == 'conductivity':
            cbar.set_label('Conductivity') 
        plt.axis(False)

    def get_aem_conductivity_plot(self,reg, conductivity_max_lim = None):

        self.get_aem_values()
        aemres = self.gdfres.copy()
        aemres=aemres.dropna(subset=['Resistivity'])

        if conductivity_max_lim is not None:
            aemres = aemres[aemres.Resistivity<=conductivity_max_lim]

        fig = plt.figure(figsize=(10, 10))
        reg.plot(label = 'Kaweah',facecolor = 'none', edgecolor = 'black', lw = .5, zorder = 10)
        out = plt.scatter(
            aemres['geometry'].x, aemres['geometry'].y, c=aemres.Resistivity, 
            s=10, 
            cmap='turbo',
            zorder = 1
        )
        plt.xlabel("Easting (m)")
        plt.ylabel("Northing (m)")
        cbar = plt.colorbar(out, fraction=0.03)
        cbar.set_label('Conductivity') 
        plt.axis(False)

    def get_combined_aem_two_regions(self, datareg2, conductivity_max_lim = None):
        """
        Combine aem data from two regions into one 
        Arg:
        datareg2: region 2 data
        conductivity_max_lim: assigning a max conductivity value below which data are extracted
        """

        self.get_aem_values()
        aemres = self.gdfres.copy()
        aemres = pd.concat([aemres, datareg2.get_aem_values()])

        self.aemres_tworegs=aemres.dropna(subset=['Resistivity'])

        if conductivity_max_lim is not None:
            self.aemres_tworegs = self.aemres_tworegs[self.aemres_tworegs.Resistivity<=conductivity_max_lim]

        return self.aemres_tworegs


    def get_scatter_aem_conductivity_tworegs(self,datareg2,reg, conductivity_max_lim = None, vmax_in = None):

        self.get_combined_aem_two_regions(self, datareg2 = datareg2, conductivity_max_lim = conductivity_max_lim)

        if vmax_in is None:
            vmax = self.aemres_tworegs.Resistivity.max()
        else:
            vmax = vmax_in

        fig = plt.figure(figsize=(10, 10))
        reg.plot(label = 'Kaweah',facecolor = 'none', edgecolor = 'black', lw = .5, zorder = 10)
        out = plt.scatter(
            self.aemres_tworegs['geometry'].x, self.aemres_tworegs['geometry'].y, c=self.aemres_tworegs.Resistivity, 
            s=10, 
            vmax = vmax,
            cmap='turbo',
            zorder = 1
        )
        plt.xlabel("Easting (m)")
        plt.ylabel("Northing (m)")
        cbar = plt.colorbar(out, fraction=0.03)
        cbar.set_label('Conductivity') 
        plt.axis(False)


    def get_scatter_aem_polut_in_bufferzone(self, datareg2 = None, yaxis_lim = None, xaxis_lim = None):
        """
        scatter plot of aem value and sagbi values spatially averaged over buffer zone
        
        Arg:
        regbound: region boundary for plot
        datareg2: aem model for additional region
        xaxis_lim, yaxis_lim: x-axis and y-axis max limits
        """
        self.get_aem_mean_in_well_buffer(datareg2 = datareg2)
        
        fig, ax = plt.subplots(figsize=(7, 7))
        plt.scatter(self.aem_wq_buff_aemmean['Resistivity'],self.aem_wq_buff_aemmean['VALUE'],
                    color = 'red', s = 1)
        plt.ylim([0,1000])
        plt.xlim([0,0.9])
        plt.xlabel(f'Mean {self.aem_value_type} inside buffer', fontsize=20)
        plt.ylabel('Nitrate concentration [mg/l]', fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=17)

        if xaxis_lim is not None:
            plt.xlim([0, xaxis_lim])
        if yaxis_lim is not None:
            plt.ylim([0,yaxis_lim])

        plt.show()