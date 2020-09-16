#!/usr/bin/env python
# coding: utf-8

# # Clustering of Zimnat's Data for Feature Engineering

# In[1]:


# Import libraries necessary for this project
import pandas as pd
import numpy as np
from IPython.display import display # Allows the use of display() for DataFrames

# Show matplotlib plots inline (nicely formatted in the notebook)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Load the wholesale customers dataset
try:
    data = pd.read_csv('C:/Users/Badiah/Desktop/training_data.csv')
    data.drop(['occupation_code','branch_code','marital_status','occupation_category_code','sex',
               'birth_year','ID','join_date','join_year'], axis = 1, inplace = True)
    print ("Insurance dataset has {} samples with {} features each.".format(*data.shape))
except:
    print ("Dataset could not be loaded. Is the dataset missing?")


# In[3]:


data.head(5)


# import itertools
# 
# # Select the indices for data points you wish to remove
# outliers_lst  = []
# 
# # For each feature find the data points with extreme high or low values
# for feature in numerical_cols:
#     # Calculate Q1 (25th percentile of the data) for the given feature
#     Q1 = np.percentile(data.loc[:, feature], 25)
# 
#     # Calculate Q3 (75th percentile of the data) for the given feature
#     Q3 = np.percentile(data.loc[:, feature], 75)
# 
#     # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
#     step = 1.5 * (Q3 - Q1)
# 
#     # Display the outliers 
#     print("Data points considered outliers for the feature '{}':".format(feature))
# 
#     # The tilde sign ~ means not
#     # So here, we're finding any points outside of Q1 - step and Q3 + step
#     outliers_rows = data.loc[~((data[feature] >= Q1 - step) & (data[feature] <= Q3 + step)), :]
#     # display(outliers_rows)
# 
#     outliers_lst.append(list(outliers_rows.index))
# 
#     outliers = list(itertools.chain.from_iterable(outliers_lst))
# 
# # List of unique outliers
# # We use set()
# # Sets are lists with no duplicate entries
# uniq_outliers = list(set(outliers))
# 
# # List of duplicate outliers
# dup_outliers = list(set([x for x in outliers if outliers.count(x) > 1]))
# 
# print ('Outliers list:\n', uniq_outliers)
# print ('Length of outliers list:\n', len(uniq_outliers))
# 
# print ('Duplicate list:\n', dup_outliers)
# print ('Length of duplicates list:\n', len(dup_outliers))
# 
# # Remove duplicate outliers
# # Only 5 specified
# data = data.drop(data.index[dup_outliers]).reset_index(drop = True)
# 
# # Original Data 
# print ('Original shape of data:\n', data.shape)
# # Processed Data
# print ('New shape of data:\n', data.shape)

# In[4]:


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

# Select columns 
numerical_cols = [cname for cname in data.columns if data[cname].dtype in ['int64', 'float64']]
categorical_cols = [cname for cname in data.columns if data[cname].dtype == "object"]

cols = {"numerical_cols":numerical_cols,"categorical_cols":categorical_cols}

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])

# Preprocessing for numerical data
numeric_transformer = Pipeline(steps=[
     ('imputer', SimpleImputer(strategy='median')),
     ('scaler', StandardScaler())])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer( transformers=[
   ('num', numeric_transformer, cols['numerical_cols']),
   ('cat', categorical_transformer, cols['categorical_cols'])]) 

data = pd.DataFrame(preprocessor.fit_transform(data))


# ### Data Exploration 
# #### Explore the data through visualizations to understand how each feature is related to the others. You will observe a statistical description of the dataset, consider the relevance of each feature, and select a few sample data points from the dataset which you will track through the project.

# In[5]:


data.head (5)


# In[6]:


# Display a description of the dataset
data.describe()


# In[7]:


# Check for Null Values
data.isnull().values.any()


# In[8]:


data.isnull().sum()


# In[9]:


# Checking for rows that have NAN -INF 

data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]


# ### Implementation: Selecting Samples
# #### To get a better understanding of the customers and how their data will transform through the analysis, it would be best to select a few sample data points and explore them in more detail.

# #### Hence we'll be choosing:
# #24: "RVSZ-K6QO Pair"
# #33: "JZ9D-J9JW Pair"
# #18:  "Neither of the 2 pairs above^" 

# In[10]:


# Select three indices of your choice you wish to sample from the dataset
indices = [0,4,18]

# Create a DataFrame of the chosen samples
# .reset_index(drop = True) resets the index from 0, 1 and 2 instead of 2,500, 5,000 and 7,500 
samples = pd.DataFrame(data.loc[indices], columns = data.columns).reset_index(drop = True)
print ("Chosen samples of training dataset:")
display(samples)


# ### Implementation: Feature Relevance

# # Imports
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor

# ### Data Preprocessing
# #### Preprocess the data to create a better representation of customers by performing a scaling on the data and detecting (and optionally removing) outliers. Preprocessing data is often times a critical step in assuring that results you obtain from your analysis are significant and meaningful.

# This indicates how normalization is required to make the data features normally distributed as clustering algorithms require them to be normally distributed.
# 
# ### Feature Scaling
# If data is not normally distributed, especially if the mean and median vary significantly (indicating a large skew), it is most often appropriate to apply a non-linear scaling — particularly for financial data. One way to achieve this scaling is by using a Box-Cox test, which calculates the best power transformation of the data that reduces skewness. A simpler approach which can work in most cases would be applying the natural logarithm.

# In[11]:


data.head


# In[12]:


# Scale the 3 samples data using the natural logarithm
scaled_samples = StandardScaler().fit_transform(samples) 


# ### Observation
# After applying a natural logarithm scaling to the data, the distribution of each feature should appear much more normal. For any pairs of features you may have identified earlier as being correlated, observe here whether that correlation is still present (and whether it is now stronger or weaker than before).

# In[13]:


# Display the transformed sample data
display(scaled_samples)


# In[14]:


pd.DataFrame(data)

#pd.DataFrame(scaled_data,columns=['P5DA', 'RIBP', '8NN1', '7POT', '66FJ','GYSR', 'SOP4',
 #      'RVSZ', 'PYUQ', 'LJR9', 'N2MW', 'AHXO', 'BSTQ', 'FM3X', 'K6QO', 'QBOL',
  #     'JWFN', 'JZ9D', 'J9JW', 'GHYX', 'ECY3'])


# In[15]:


import itertools


# ### Feature Transformation
# We'll use Principal Component Analysis (PCA) to draw conclusions about the underlying structure of the customer data. Since using PCA on a dataset calculates the dimensions which best maximize variance, we will find which compound combinations of features best describe customers.
# 
# Implementation: PCA
# Now that the data has been scaled to a more normal distribution and has had any necessary outliers removed, we can now apply PCA to the good_data to discover which dimensions about the data best maximize the variance of features involved. In addition to finding these dimensions, PCA will also report the explained variance ratio of each dimension — how much variance within the data is explained by that dimension alone.
# 
# Note that a component (dimension) from PCA can be considered a new "feature" of the space, however it is a composition of the original features present in the data.

# In[16]:


from sklearn.decomposition import PCA
from sklearn import preprocessing


# In[17]:


# Apply PCA by fitting the good data with the same number of dimensions as features
# Instantiate
pca = PCA(n_components=21)
# Fit
pca.fit(data)
# Transform the sample log-data using the PCA fit above
pca_samples = pca.transform(scaled_samples)

pca_results = pca.transform(data)

# Generate PCA results plot
# pca_results = rs.pca_results(data, pca)


# In[18]:


#DataFrame of results
display(pca_results)


# In[19]:


ps = pd.DataFrame(pca_results)
ps.head()


# How has the log-transformed sample data changed after having a PCA transformation applied to it in six dimensions?
# Observe the numerical value for the first four dimensions of the sample points. Consider if this is consistent with your initial interpretation of the sample points.

# In[20]:


# NOTE!!! Display sample scaled_data after having a PCA transformation applied
display(pd.DataFrame(np.round(pca_results, 4),columns=['P5DA', 'RIBP', '8NN1', '7POT', '66FJ','GYSR', 'SOP4',
       'RVSZ', 'PYUQ', 'LJR9', 'N2MW', 'AHXO', 'BSTQ', 'FM3X', 'K6QO', 'QBOL',
       'JWFN', 'JZ9D', 'J9JW', 'GHYX', 'ECY3']))


# In[21]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#Cumulative explained variance should add to 1 NOTE?!!!
#display(pca_results['Explained Variance'].cumsum())

# Draw a scree plot and a PCA plot
 
#The following code constructs the Scree plot
per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
 
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()


# In[22]:


#the following code makes a fancy looking plot using PC1 and PC2
pca_df = pd.DataFrame(pca_results, columns=labels)

#pca_df = pd.DataFrame(pca_results, , index=[*wt, *ko], columns=labels)
 
plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title('My PCA Graph')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))
 
for sample in pca_df.index:
    plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))
 
plt.show()


# In[23]:


# Determine which data point had the biggest influence on PC1 get the name of the top 10 measurements that contribute most to pc1.
#first, get the loading scores
loading_scores = pd.Series(pca.components_[0])
#loading_scores = pd.Series(pca.components_[0], index=genes)

# now sort the loading scores based on their magnitude
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
 
# get the names of the top 10 genes
top_7_variables = sorted_loading_scores[0:7].index.values

## print the gene names and their scores (and +/- sign)
print(loading_scores[top_7_variables]) #<span id="mce_SELREST_start" style="overflow:hidden;line-height:0;"></span>


# columns=['P5DA', 'RIBP', '8NN1', '7POT', '66FJ','GYSR', 'SOP4',
#        'RVSZ', 'PYUQ', 'LJR9', 'N2MW', 'AHXO', 'BSTQ', 'FM3X', 'K6QO', 'QBOL',
#        'JWFN', 'JZ9D', 'J9JW', 'GHYX', 'ECY3']))

# In[ ]:


from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
tocluster = pd.DataFrame(ps[[1,2]])
print (tocluster.shape)
print (tocluster.head())

fig = plt.figure(figsize=(8,8))
plt.plot(tocluster[1], tocluster[2], 'o', markersize=2, color='blue', alpha=0.25, label='class1')

plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.show()


# In[ ]:





# ### Dimensionality Reduction
# When using Principal Component Analysis (PCA), one of the main goals is to reduce the dimensionality of the data — in effect, reducing the complexity of the problem. Dimensionality reduction comes at a cost: Fewer dimensions used implies less of the total variance in the data is being explained.
# Because of this, the cumulative explained variance ratio is extremely important for knowing how many dimensions are necessary for the problem. Additionally, if a signifiant amount of variance is explained by only two or three dimensions, the reduced data can be visualized afterwards.

# In[ ]:


# Apply PCA by fitting the good data with only two dimensions
# Instantiate
pca = PCA(n_components=3)
pca.fit(data)

# Transform the good data using the PCA fit above
reduced_data = pca.transform(data)

# Transform the sample log-data using the PCA fit above
pca_samples = pca.transform(samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2', 'Dimension 3'])


# Run the code below to see how the log-transformed sample data has changed after having a PCA transformation applied to it using only two dimensions. Observe how the values for the first two dimensions remains unchanged when compared to a PCA transformation in six dimensions.

# In[ ]:


# Display sample scaled-data after applying PCA transformation in two dimensions
display(pd.DataFrame(np.round(reduced_data, 4), columns = ['Dimension 1', 'Dimension 2', 'Dimension 3']))


# ## Clustering
# We will choose to use either a K-Means clustering algorithm or a Gaussian Mixture Model clustering algorithm to identify the various customer segments hidden in the data.
# 
# We'll then recover specific data points from the clusters to understand their significance by transforming them back into their original dimension and scale.

# ### Creating Clusters
# Depending on the problem, the number of clusters that you expect to be in the data may already be known. When the number of clusters is not known a priori, there is no guarantee that a given number of clusters best segments the data, since it is unclear what structure exists in the data — if any.
# However, we can quantify the "goodness" of a clustering by calculating each data point's silhouette coefficient. The silhouette coefficient for a data point measures how similar it is to its assigned cluster from -1 (dissimilar) to 1 (similar). Calculating the mean silhouette coefficient provides for a simple scoring method of a given clustering.

# In[ ]:


# Imports
from sklearn.mixture import GaussianMixture as GMM #formerly GMM
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[ ]:


# Create range of clusters 
range_n_clusters = list(range(2,8))
print(range_n_clusters)


# #### GMM Implementation

# In[ ]:


# Loop through clusters
for n_clusters in range_n_clusters:
    # Apply your clustering algorithm of choice to the reduced data 
    clusterer = GMM(n_components=n_clusters).fit(reduced_data)

    # Predict the cluster for each data point
    preds = clusterer.predict(reduced_data)
    
    # Find the cluster centers
    centers = clusterer.means_

    # Predict the cluster for each transformed sample data point
    sample_preds = clusterer.predict(pca_samples)

    # Calculate the mean silhouette coefficient for the number of clusters chosen
    score = silhouette_score(reduced_data, preds, metric='mahalanobis')
    print ("For n_clusters = {}. The average silhouette_score is : {}".format(n_clusters, score))


# In[ ]:


X=reduced_data #NOTE!!!
lowest_bic = np.infty
bic = []
n_components_range = range(1, 5)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a mixture of Gaussians with EM
        gmm = GMM(n_components=n_components, covariance_type=cv_type)
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm


# In[ ]:


# Probability that any point belongs to the given cluster

probs = gmm.predict_proba(X)
print(probs[:200].round(4))


# In[ ]:


#size = 50 * probs.max(1) ** 2  # square emphasizes differences
#plt.scatter(X[:, 1], c=labels, cmap='viridis', s=size);


# In[ ]:





# KMeans Implementation

# In[ ]:


# Loop through clusters
for n_clusters in range_n_clusters:
    # Apply your clustering algorithm of choice to the reduced data 
    clusterer = KMeans(n_clusters=n_clusters).fit(reduced_data)

    # Predict the cluster for each data point
    preds = clusterer.predict(reduced_data)

    # Find the cluster centers
    centers = clusterer.cluster_centers_

    # Predict the cluster for each transformed sample data point
    sample_preds = clusterer.predict(pca_samples)

    # Calculate the mean silhouette coefficient for the number of clusters chosen
    score = silhouette_score(reduced_data, preds, metric='euclidean')
    print ("For n_clusters = {}. The average silhouette_score is : {}".format(n_clusters, score))


# Distance Metric: The Silhouette Coefficient is calculated using the mean intra-cluster distance and the mean nearest-cluster distance for each sample. Therefore, it makes sense to use the same distance metric here as the one used in the clustering algorithm. This is Euclidean for KMeans and Mahalanobis for general GMM.

# ### Cluster Visualization
# Once you've chosen the optimal number of clusters for your clustering algorithm using the scoring metric above, you can now visualize the results by executing the code block below.

# In[ ]:


# Extra code because we ran a loop on top and this resets to what we want
clusterer = GMM(n_components=2).fit(reduced_data) #clusterer = KMeans(n_clusters=4,random_state=42).fit(tocluster)
centers = clusterer.means_  #centers = clusterer.cluster_centers_
preds = clusterer.predict(reduced_data)
sample_preds = clusterer.predict(pca_samples)


# In[ ]:


# Display the results of the clustering from implementation
# rs.cluster_results(reduced_data, preds, centers, pca_samples)
print(centers)


# In[ ]:


print (preds[0:10])


# ### How our clusters appear

# In[ ]:


import matplotlib
fig = plt.figure(figsize=(2,2))
colors = ['blue','red']
colored = [colors[k] for k in preds]
print (colored[0:10])


# In[ ]:


plt.scatter(tocluster[1],tocluster[2],  color = colored)
for ci,c in enumerate(centers):
    plt.plot(c[0],c[1], 'o', markersize=8, color='', alpha=0.9, label=''+str(ci))

plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.show()


# ### Data Recovery
# Each cluster present in the visualization above has a central point. These centers (or means) are not specifically data points from the data, but rather the averages of all the data points predicted in the respective clusters.
# For the problem of creating customer segments, a cluster's center point corresponds to the average customer of that segment. Since the data is currently reduced in dimension and scaled by a logarithm, we can recover the representative customer spending from these data points by applying the inverse transformations.

# In[ ]:


# Inverse transform the centers
log_centers = pca.inverse_transform(centers)

# Exponentiate the centers
true_centers = np.exp(log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.columns)
true_centers.index = segments
display(true_centers)


# ### Display data in Clusters & Rename IDs as Clusters / Feature Engineering??

# In[ ]:


# Display the predictions
for i, pred in enumerate(sample_preds):
    print ("Sample point", i, "predicted to be in Cluster", pred)


# ### Conclusion
# We will investigate ways that you can make use of the clustered data.
# We will consider how giving a label to each customer (which segment that customer belongs to) can provide for "additional features about the customer data".
