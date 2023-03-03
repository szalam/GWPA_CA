#%%
import sys
sys.path.insert(0,'src')

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import config

# Read dataset
data = pd.read_csv(config.data_processed / "Dataset_processed.csv")
# data = data.drop(['well_id', 'well_data_source','start_date', 'end_date'], axis=1)

data = data[['mean_nitrate','Conductivity']]
# Replace all NaN values with blank
# data = data.fillna('0')

# Remove rows with missing values
data = data.dropna()

# Define the number of clusters
n_clusters = 2

# Perform k-means clustering
kmeans = KMeans(n_clusters=n_clusters)
#%%
kmeans.fit(data)
#%%

# Create a scatter plot of the data with the clusters colored
plt.scatter(data['Conductivity'], data['mean_nitrate'], c=kmeans.labels_)
plt.show()
# %%
