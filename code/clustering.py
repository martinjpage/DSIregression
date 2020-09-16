# Import libraries necessary for this project
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.decomposition import PCA

def get_customer_cluster(preds, data_dir):

    # Import the Datasets
    train_data = pd.read_csv(data_dir)

    #Define target columns
    train_data = train_data[['P5DA', 'RIBP', '8NN1', '7POT', '66FJ','GYSR', 'SOP4', 'RVSZ', 'PYUQ', 'LJR9', 'N2MW',
                             'AHXO', 'BSTQ', 'FM3X','K6QO', 'QBOL', 'JWFN', 'JZ9D', 'J9JW', 'GHYX', 'ECY3']]

    #train_data = StandardScaler().fit_transform(train_data)

    # Apply PCA by fitting the good data with the same number of dimensions as features
    # Instantiate
    pca = PCA(n_components=21)
    # Fit
    pca.fit(train_data)

    pca_results = pca.transform(train_data)
    ps = pd.DataFrame(pca_results) #Dataframe of results

    # Apply PCA by fitting the good data with only two dimensions
    # Instantiate
    pca = PCA(n_components=4)
    pca.fit(train_data)

    # Transform the good data using the PCA fit above
    reduced_data = pca.transform(train_data)

    # Create a DataFrame for the reduced data
    reduced_data = pd.DataFrame(reduced_data, columns=['Dimension 1',
                                                       'Dimension 2',
                                                       'Dimension 3',
                                                       'Dimension 4'])

    # Extra code because we ran a loop on top and this resets to what we want
    # ### Data Recovery
    # Each cluster present in the visualization above has a central point.
    # These centers (or means) are not specifically data points from the data,
    # but rather the averages of all the data points predicted in the respective clusters.
    # For the problem of creating customer segments, a cluster's center point
    # corresponds to the average customer of that segment. Since the data is
    # currently reduced in dimension and scaled by a logarithm, we can recover the
    # representative customer spending from these data points by applying the inverse
    # transformations.

    # Create index_list
    index = list(train_data.index)

    clusterer = GMM(n_components=8).fit(reduced_data)

    # Updated top 4 products representing 70% of the variance
    reduced_data['preds'] = train_data[['RVSZ', 'PYUQ','K6QO', 'QBOL']].apply\
        (lambda x: clusterer.predict([x])[0], axis=1)
    # print(reduced_data['preds'])

    return reduced_data['preds'].to_numpy()