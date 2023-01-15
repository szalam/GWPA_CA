#%%
import sys
sys.path.insert(0,'src')

import config
import pandas as pd

import matplotlib.pyplot as plt

# %%

class vismod():

    def __init__(self,df) -> None:
        self.df = df

    def get_scatter_aem_polut(self, xcolm_name, ycolm_name, yaxis_lim = None, xaxis_lim = None, YlabelC ='tmp', XlabelC ='tmp' ,yunitylabel ='tmp', xunitylabel ='tmp'):
        """
        Get scatter plot of aem values and pollutant concentration

        Arg:
        
        yaxis_lim; xaxis_lim: None = uses max value, user input= uses max value
        gwpaflag: None = outside gwpa, 1 = insude gwpa, both = both, All: plot all
        YlabelC: yaxis label
        unitylabel: y-axis unit
        """

        fig, ax = plt.subplots(figsize=(7, 7))
        
        ax.scatter(self.df[xcolm_name],self.df[ycolm_name],color = 'red', s = 1)
        
        if xaxis_lim is not None:
            plt.xlim([0, xaxis_lim])

        if yaxis_lim is None:
            plt.ylim([0,self.df[ycolm_name].max()])
        else:
            plt.ylim([0,yaxis_lim])

        plt.xlabel(f'{XlabelC} {xunitylabel}', fontsize=20)

        plt.ylabel(f'{YlabelC} {yunitylabel}', fontsize=20)
        plt.tick_params(axis='both', which='major', labelsize=17)
        plt.show()


