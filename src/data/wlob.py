import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from shapely.geometry import Point
import wlfun as wl

sys.path.insert(0,'src')
import config


class wlob():

    def __init__(self, well_stations,well_measurements,well_perforations,CA_map, CV_map, CC_map):

        self.stations = well_stations
        self.measurements = well_measurements
        self.perforations = well_perforations
        self.CA_map = CA_map
        self.CV_map = CV_map
        self.CC_map = CC_map


    def get_cvhm_lyr(self):
        zone_map = gpd.read_file(config.shapefile_dir / 'ZONE.shp')
        zone_map.to_crs(epsg=config.crs_latlon, inplace=True)
        zone_map = zone_map.sort_values(['ROW', 'COLUMN_'], ascending=[True, True])

        # %%
        ### not the most pythonic way of doing this...
        ### numpy's flatten is also row-major order, which aligns with what we did with the zone information
        layer1 = np.genfromtxt(config.shapefile_dir / 'FIN_THK_L1C.txt', delimiter=',').flatten()
        layer2 = np.genfromtxt(config.shapefile_dir / 'FIN_THK_L2C.txt', delimiter=',').flatten()
        layer3 = np.genfromtxt(config.shapefile_dir / 'FIN_THK_L3.txt', delimiter=',').flatten()
        layer4 = np.genfromtxt(config.shapefile_dir / 'FIN_THK_L4.txt', delimiter=',').flatten()
        layer5 = np.genfromtxt(config.shapefile_dir / 'FIN_THK_L5.txt', delimiter=',').flatten()
        layer6 = np.genfromtxt(config.shapefile_dir / 'FIN_THK_L6.txt', delimiter=',').flatten()
        layer7 = np.genfromtxt(config.shapefile_dir / 'FIN_THK_L7.txt', delimiter=',').flatten()
        layer8 = np.genfromtxt(config.shapefile_dir / 'FIN_THK_L8.txt', delimiter=',').flatten()
        layer9 = np.genfromtxt(config.shapefile_dir / 'FIN_THK_L9.txt', delimiter=',').flatten()
        layer10 = np.genfromtxt(config.shapefile_dir / 'FIN_THK_L10.txt', delimiter=',').flatten()
        # %%
        zone_map['layer1_thk'] = layer1
        zone_map['layer2_thk'] = layer2
        zone_map['layer3_thk'] = layer3
        zone_map['layer4_thk'] = layer4
        zone_map['layer5_thk'] = layer5
        zone_map['layer6_thk'] = layer6
        zone_map['layer7_thk'] = layer7
        zone_map['layer8_thk'] = layer8
        zone_map['layer9_thk'] = layer9
        zone_map['layer10_thk'] = layer10
        # %%
        zone_map['unconfined_bot'] = zone_map['layer1_thk'] + zone_map['layer2_thk'] + zone_map['layer3_thk']

        return zone_map


    def well_filter_qa(self):
        """
        Remove well station with certain QA comments
        Arg:
        well_measurements: Well measurement data
        """
        bad_descs = ['Pumped recently', 'Caved or deepened', 'Nearby pump operating', 'Pumping',
                    "Can't get tape in casing", 'Well has been destroyed', 'Measurement Discontinued',
                    'Recharge or surface water effects near well', 'Dry well', 'Flowing artesian well',
                    'Recently flowing', 'Unable to locate well', 'Temporarily inaccessible',
                    'Oil or foreign substance in casing', 'Pump house locked', 'Flowing',
                    'Tape hung up', 'Casing leaking or wet', 'Nearby flowing',
                    'Nearby recently flowing']

        self.measurements_filter_qa = self.measurements[~self.measurements["wlm_qa_desc"].isin(bad_descs)]
        
        return self.measurements_filter_qa

    def well_meas_perfor_combine(self):

        # Drop monitoring program column.
        self.stations = self.stations.drop('monitoring_program', 1)
        # self.stations.info()

        # print(well_stations["well_use"].unique())
        # self.perforations.info()
        # Drop duplicate well site information
        self.stations = self.stations.drop_duplicates(subset=['site_code'])

        added_column_list = list(self.stations.columns)
        # print(added_column_list)
        added_column_list.extend(['TOP_PRF_INT_0', 'BOT_PRF_INT_0',
                                'TOP_PRF_INT_1', 'BOT_PRF_INT_1',
                                'TOP_PRF_INT_2', 'BOT_PRF_INT_2',
                                'TOP_PRF_INT_3', 'BOT_PRF_INT_3',
                                'TOP_PRF_INT_4', 'BOT_PRF_INT_4',
                                'TOP_PRF_INT_5', 'BOT_PRF_INT_5',
                                'TOP_PRF_INT_6', 'BOT_PRF_INT_6',
                                'TOP_PRF_INT_7', 'BOT_PRF_INT_7',
                                'TOP_PRF_INT_8', 'BOT_PRF_INT_8',
                                'TOP_PRF_INT_9', 'BOT_PRF_INT_9'])
        # print(added_column_list)
        self.stations_perforations = self.stations.reindex(columns = added_column_list)
        
        # Combining well perforation with well station data
        for index, row in self.perforations.iterrows():
            station_index = self.stations_perforations.index[
                self.stations_perforations['site_code'] == row['site_code']].tolist()[0]

            for i in range(10):
                if np.isnan(self.stations_perforations.at[station_index,f'TOP_PRF_INT_{i}']):
                    self.stations_perforations.at[station_index, f'TOP_PRF_INT_{i}'] = row["top_prf_int"]
                    self.stations_perforations.at[station_index, f'BOT_PRF_INT_{i}'] = row["bot_prf_int"]
                

        # Merge well perforation with well measurements
        well_msp = self.measurements.merge(self.stations_perforations, how='left', left_on='site_code', right_on='site_code')
        
        return well_msp


    def get_well_from_use_region(self, filter_qa = None, well_use = None, all_wells = None, CV_map = None, CC_map = None):
        """
        
        """
        if all_wells is None:
            if filter_qa is not None:
                wells_sel = self.stations_perforations[(self.stations_perforations["well_use"] == well_use) 
                                        & (~self.stations_perforations["wcr_no"].isna()) & 
                                        (self.stations_perforations["wcr_no"] != 'YES') &
                                        (~self.stations_perforations["TOP_PRF_INT_0"].isna())]
            if filter_qa is None:
                wells_sel = self.stations_perforations[(self.stations_perforations["well_use"] == well_use)]

            well_points = [Point(xy) for xy in zip(wells_sel["longitude"], wells_sel["latitude"])]
            well_sites = gpd.GeoDataFrame(wells_sel, crs=config.crs_latlon, geometry=well_points)

        if all_wells is not None:
            well_points = [Point(xy) for xy in zip(self.stations_perforations["longitude"], 
                                           self.stations_perforations["latitude"])]
            well_sites = gpd.GeoDataFrame(self.stations_perforations, crs=config.crs_latlon, geometry=well_points)


        if CV_map is not None:
            region_sites = well_sites.sjoin(CV_map)
        if CC_map is not None:
            region_sites = well_sites.sjoin(CC_map)

        if 'index_right' in region_sites.columns:
            region_sites = region_sites.drop('index_right', 1)

        return region_sites


    def get_shallow_wells(self,zone_map, cv_all_sites,cv_obs_sites, well_type = 'all'):
        # Process zones
        cv_zone = zone_map.sjoin(self.CV_map)
        cv_all_sites = cv_all_sites.sjoin(self.CV_map)
        cv_zone['geometry'] = cv_zone['geometry'].to_crs(3857).centroid # web mercator
        cv_zone.to_crs(config.crs_latlon, inplace=True)
        nearest_zone = wl.ckdnearest(cv_obs_sites, cv_zone)
        nearest_zone_all = wl.ckdnearest(cv_all_sites, cv_zone)

        if well_type == 'all':
            shallow_wells = nearest_zone_all[nearest_zone_all["well_depth"] < nearest_zone_all["unconfined_bot"]*3.281]
        if well_type == 'obs':
            shallow_wells = nearest_zone[nearest_zone["well_depth"] < nearest_zone["unconfined_bot"]*3.281]
        
        return shallow_wells


    def get_wells_by_aquifer(self,cv_obs_sites,shallow_wells,st_list):
        
        fig, ax = plt.subplots(figsize=(15,15))
        self.CA_map.boundary.plot(ax=ax, color='k', zorder=1)
        self.CV_map.boundary.plot(ax=ax, color='k', zorder=1)

        #CC_map.plot(ax=ax, color='r')
        cv_obs_sites.plot(ax=ax, zorder=2, label='semi-confined or confined wells')
        shallow_wells.plot(ax=ax, color='g', zorder=3, label='unconfined monitoring wells')

        for index, stn_sel in enumerate(st_list):
            shallow_wells[shallow_wells["site_code"]==stn_sel].plot(ax=ax, color='r', label=f"{stn_sel}",zorder=10)
        ax.set_xlabel('Longitude', fontsize=20)
        ax.set_ylabel('Latitude', fontsize=20)
        ax.set_title('Monitoring Well Locations in California', fontsize=24)
        ax.grid(zorder=0)

        plt.legend(fontsize=16)


    def get_well_plot_by_type(self, cv_sites, well_label):
        fig, ax = plt.subplots(figsize=(15,15))
        self.CA_map.boundary.plot(ax=ax, color='k', zorder=1)
        self.CV_map.boundary.plot(ax=ax, color='k', zorder=1)
        self.CC_map.plot(ax=ax, color='r')
        cv_sites.plot(ax=ax, zorder=2)

        ax.set_xlabel('Longitude', fontsize=20)
        ax.set_ylabel('Latitude', fontsize=20)
        
        ax.set_title(f'{well_label} Well Locations in California', fontsize=24)
        ax.grid(zorder=0)