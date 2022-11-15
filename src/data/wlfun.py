import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

from shapely.geometry import Point
import matplotlib as mpl
from arcgis.gis import GIS
import pykrige.uk
from tqdm.notebook import tqdm
import datetime as datetime
from sklearn.linear_model import LinearRegression
from scipy.spatial import cKDTree
from pygeostat.data import iotools
import scipy
import cmcrameri.cm as cmc

sys.path.insert(0,'src')
import config

def ckdnearest(gdA, gdB):
    '''scipy's binary tree search for finding nearest neighbor points between two geodataframes'''
    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)
    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            gdB_nearest,
            pd.Series(dist, name='dist')
        ], 
        axis=1)

    return gdf




def get_cvhm_lyr():
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

def well_filter_bad(well_measurements):
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

    well_measurements = well_measurements[~well_measurements["wlm_qa_desc"].isin(bad_descs)]
    
    return well_measurements

def well_meas_perfor_combine(well_stations,well_measurements,well_perforations):
    # Drop monitoring program column.
    well_stations = well_stations.drop('monitoring_program', 1)
    well_stations.info()

    # print(well_stations["well_use"].unique())
    well_perforations.info()
    # Drop duplicate well site information
    well_stations = well_stations.drop_duplicates(subset=['site_code'])

    added_column_list = list(well_stations.columns)
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
    well_stations_perforations = well_stations.reindex(columns = added_column_list)
    
    # Combining well perforation with well station data
    for index, row in well_perforations.iterrows():
        station_index = well_stations_perforations.index[
            well_stations_perforations['site_code'] == row['site_code']].tolist()[0]

        for i in range(10):
            if np.isnan(well_stations_perforations.at[station_index,f'TOP_PRF_INT_{i}']):
                well_stations_perforations.at[station_index, f'TOP_PRF_INT_{i}'] = row["top_prf_int"]
                well_stations_perforations.at[station_index, f'BOT_PRF_INT_{i}'] = row["bot_prf_int"]
            

    # Merge well perforation with well measurements
    well_msp = well_measurements.merge(well_stations_perforations, how='left', left_on='site_code', right_on='site_code')
    
    return well_msp

# Get resistivity value using coordinate. 
def zvalue_from_xy(xx,yy,values, coord):
    """
    Uses coordinate information and values as np meshgrid to extract resistivity value for given coordinate
    Arg:
    xx, yy: x and y coordinate as np array
    values: z values for given coordinate
    coord: coordinate for which data needs to be extracted
    """
    coord_idx = np.argwhere((xx==coord[0]) & (yy==coord[1]))[0]
    
    return values[coord_idx[1],coord_idx[0]]
#==================================
# Interpolation
#==================================

class tree(object):
    """
    Compute the score of query points based on the scores of their k-nearest neighbours,
    weighted by the inverse of their distances.
    @reference:
    https://en.wikipedia.org/wiki/Inverse_distance_weighting
    Arguments:
    ----------
        X: (N, d) ndarray
            Coordinates of N sample points in a d-dimensional space.
        z: (N,) ndarray
            Corresponding scores.
        leafsize: int (default 10)
            Leafsize of KD-tree data structure;
            should be less than 20.
    Returns:
    --------
        tree instance: object
    Example:
    --------
    # 'train'
    idw_tree = tree(X1, z1)
    # 'test'
    spacing = np.linspace(-5., 5., 100)
    X2 = np.meshgrid(spacing, spacing)
    X2 = np.reshape(X2, (2, -1)).T
    z2 = idw_tree(X2)
    See also:
    ---------
    demo()
    """
    def __init__(self, X=None, z=None, leafsize=10):
        if not X is None:
            self.tree = cKDTree(X, leafsize=leafsize )
        if not z is None:
            self.z = np.array(z)

    def fit(self, X=None, z=None, leafsize=10):
        """
        Instantiate KDtree for fast query of k-nearest neighbour distances.
        Arguments:
        ----------
            X: (N, d) ndarray
                Coordinates of N sample points in a d-dimensional space.
            z: (N,) ndarray
                Corresponding scores.
            leafsize: int (default 10)
                Leafsize of KD-tree data structure;
                should be less than 20.
        Returns:
        --------
            idw_tree instance: object
        Notes:
        -------
        Wrapper around __init__().
        """
        return self.__init__(X, z, leafsize)

    def __call__(self, X, k=6, eps=1e-6, p=2, regularize_by=1e-9):
        """
        Compute the score of query points based on the scores of their k-nearest neighbours,
        weighted by the inverse of their distances.
        Arguments:
        ----------
            X: (N, d) ndarray
                Coordinates of N query points in a d-dimensional space.
            k: int (default 6)
                Number of nearest neighbours to use.
            p: int or inf
                Which Minkowski p-norm to use.
                1 is the sum-of-absolute-values "Manhattan" distance
                2 is the usual Euclidean distance
                infinity is the maximum-coordinate-difference distance
            eps: float (default 1e-6)
                Return approximate nearest neighbors; the k-th returned value
                is guaranteed to be no further than (1+eps) times the
                distance to the real k-th nearest neighbor.
            regularise_by: float (default 1e-9)
                Regularise distances to prevent division by zero
                for sample points with the same location as query points.
        Returns:
        --------
            z: (N,) ndarray
                Corresponding scores.
        """
        self.distances, self.idx = self.tree.query(X, k, eps=eps, p=p)
        self.distances += regularize_by
        weights = self.z[self.idx.ravel()].reshape(self.idx.shape)
        mw = np.sum(weights/self.distances, axis=1) / np.sum(1./self.distances, axis=1)
        return mw

    def transform(self, X, k=6, p=2, eps=1e-6, regularize_by=1e-9):
        """
        Compute the score of query points based on the scores of their k-nearest neighbours,
        weighted by the inverse of their distances.
        Arguments:
        ----------
            X: (N, d) ndarray
                Coordinates of N query points in a d-dimensional space.
            k: int (default 6)
                Number of nearest neighbours to use.
            p: int or inf
                Which Minkowski p-norm to use.
                1 is the sum-of-absolute-values "Manhattan" distance
                2 is the usual Euclidean distance
                infinity is the maximum-coordinate-difference distance
            eps: float (default 1e-6)
                Return approximate nearest neighbors; the k-th returned value
                is guaranteed to be no further than (1+eps) times the
                distance to the real k-th nearest neighbor.
            regularise_by: float (default 1e-9)
                Regularise distances to prevent division by zero
                for sample points with the same location as query points.
        Returns:
        --------
            z: (N,) ndarray
                Corresponding scores.
        Notes:
        ------
        Wrapper around __call__().
        """
        return self.__call__(X, k, eps, p, regularize_by)
# %%
#=========================
# Visualization codes
#=========================

def get_cv_map(CA = None, CV = None,corcoran = None, zone_map = None):
    fig, ax = plt.subplots(figsize=(15,15))
    if CA is not None:
        CA.boundary.plot(ax=ax, color='k')
    if CV is not None:
        CV.boundary.plot(ax=ax)
    if corcoran is not None:
        corcoran.plot(ax=ax, color='r')
    if zone_map is not None:
        zone_map.boundary.plot(ax=ax, color='k')
    ax.set_title('Shapefiles for California, Central Valley, and Corcoran Clay', fontsize=24)


