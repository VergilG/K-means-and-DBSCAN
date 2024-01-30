import pandas as pd  # data visualized
import scipy as sc  # calculating
import numpy as np  # calculating
import seaborn as sns  # data visualized
import matplotlib.pyplot as plt  # generating plots
from sklearn import decomposition  # orthogonalize method
from sklearn import preprocessing  # preprocessing data
from sklearn.neighbors import NearestNeighbors  # nearest neighbor method
from sklearn.cluster import DBSCAN  # clustering method

# Data pre-processing
climateRaw = pd.read_csv('ClimateDataBasel.csv', header=None)
climateData = pd.DataFrame(climateRaw)

# Show the structure of data
print('Climate data has {} rows and {} cols.'.format(climateData.shape[0], climateData.shape[1]))

# Data Cleaning
# Fill missing values
check = np.isnan(climateData).any(axis=0)
if check.any():
    print("There are missing values in the dataset")
else:
    print("There are no missing values in the dataset")

# Feature Selection
climateCorrMatrix = np.corrcoef(climateData, rowvar=False)  # Calculate correlation

# Plot the correlation matrix by using heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(np.abs(climateCorrMatrix), cmap='coolwarm')
plt.title('Climate Correlation Matrix')
plt.savefig('ClimateCorrelation.pdf')
plt.show()

# Select the features
select_features = [2, 5, 8, 11, 14, 17]  # Mean features and sunshine duration
selected_data = climateData[select_features]

# Standardize data
preprocessedClimate = preprocessing.StandardScaler().fit_transform(selected_data)

# Find outliers
row = []
for i in range(len(preprocessedClimate)):
    for j in range(len(preprocessedClimate[i])):
        if preprocessedClimate[i][j] > 3 or preprocessedClimate[i][j] < -3:
            row.append(i)

row = list(set(row))
print("There are {} outliers.".format(len(row)))

# Normalize data
preprocessedClimate = preprocessing.MinMaxScaler().fit_transform(preprocessedClimate)

# PCA method
# With outliers
pca_preprocess = decomposition.PCA(n_components=4).fit(preprocessedClimate)
pca_preprocess_climate = pca_preprocess.transform(preprocessedClimate)

print('The explained variance ratio of each principle component with outliers is {}.'.format(
    pca_preprocess.explained_variance_ratio_))

# Without outliers
noExceptionClimate = pd.DataFrame(preprocessedClimate).drop(row)
pca_no_exception = decomposition.PCA(n_components=4).fit(noExceptionClimate)
pca_no_exception_climate = pca_no_exception.transform(noExceptionClimate)

print('The explained variance ratio of each principle component without outliers is {}.'.format(
    pca_no_exception.explained_variance_ratio_))

# Plot the PCA Data
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].scatter(pca_preprocess_climate[:, 0], pca_preprocess_climate[:, 1], marker='.')
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[0].set_title('With Outliers')

ax[1].scatter(pca_no_exception_climate[:, 0], pca_no_exception_climate[:, 1], marker='.')
ax[1].set_xlabel('PC1')
ax[1].set_ylabel('PC2')
ax[1].set_title('Without Outliers')

plt.savefig('pca_plot.pdf')
plt.show()


# Clustering
def distance(d1, d2):  # Calculating distance of two points
    sumSqar = 0
    for i in range(len(d1)):
        sumSqar = sumSqar + (d1[i] - d2[i]) ** 2

    return sumSqar ** 0.5


# K-Means Clustering
data_kmeans = pca_no_exception_climate[:, 0:3].copy()  # First 3 PCs can explain about 90% variance.

# Elbowing Method
distortions = []
for k in range(1, 11):
    centroid, distortion = sc.cluster.vq.kmeans(data_kmeans, k, iter=300)
    distortions.append(distortion)

plt.plot(distortions)
plt.show()

# Cluster data with k=2
centroids2, distortion2 = sc.cluster.vq.kmeans(data_kmeans, 2, iter=300)

group1 = np.array([])
group2 = np.array([])

for i in data_kmeans:
    if distance(i, centroids2[0, :]) < distance(i, centroids2[1, :]):
        if len(group1) == 0:
            group1 = i
        else:
            group1 = np.vstack((group1, i))
    else:
        if len(group2) == 0:
            group2 = i
        else:
            group2 = np.vstack((group2, i))

# Cluster data with k=3
centroids3, distortion3 = sc.cluster.vq.kmeans(data_kmeans, 3, iter=300)

group31 = np.array([])
group32 = np.array([])
group33 = np.array([])

for i in data_kmeans:
    d = [distance(i, centroids3[0, :]), distance(i, centroids3[1, :]), distance(i, centroids3[2, :])]
    if min(d) == d[0]:
        if len(group31) == 0:
            group31 = i
        else:
            group31 = np.vstack((group31, i))
    elif min(d) == d[1]:
        if len(group32) == 0:
            group32 = i
        else:
            group32 = np.vstack((group32, i))
    elif min(d) == d[2]:
        if len(group33) == 0:
            group33 = i
        else:
            group33 = np.vstack((group33, i))

# Plot the classified data when k = 2
_, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].scatter(group1[:, 0], group1[:, 1], c='red', marker='.')
ax[0].scatter(group2[:, 0], group2[:, 1], c='blue', marker='.')

ax[0].scatter(centroids2[:, 0], centroids2[:, 1], c='black', marker='o')

# Plot the classified data when k = 3
ax[1].scatter(group31[:, 0], group31[:, 1], c='red', marker='.')
ax[1].scatter(group32[:, 0], group32[:, 1], c='blue', marker='.')
ax[1].scatter(group33[:, 0], group33[:, 1], c='green', marker='.')

ax[1].scatter(centroids3[:, 0], centroids3[:, 1], c='black', marker='o')

plt.savefig('kmeans.pdf')

plt.show()

# DBSCAN Clustering
# k-distance plot for eps
data_DBSCAN = pca_no_exception_climate[:, 0:3].copy()

MinPts = 12  # set up the MinPts of the DBSCAN

neighbors = NearestNeighbors(n_neighbors=MinPts).fit(data_DBSCAN)  # Fit the model
distances, indices = neighbors.kneighbors(data_DBSCAN)  # Calculate the MinPtsth neighbor of each point

sortedDist = np.sort(distances[:, MinPts - 1], axis=0)  # Sort the distances

# Plot the k-distance plot
plt.Figure(figsize=(6, 4))
plt.plot(sortedDist)

plt.show()

# DBSCAN
eps = 0.082  # Set up initial parameters

dbscan = DBSCAN(eps=eps, min_samples=MinPts).fit(data_DBSCAN)  # Fit the model

labels = dbscan.labels_
cores = dbscan.core_sample_indices_

n = data_DBSCAN.shape[0]
core_point = np.zeros(n, dtype=bool)
core_point[cores] = True

unique_labels = set(labels) - {-1}  # Drop the nosies, and keep unique labels

# plot 3d scatter plot
fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111, projection='3d')

for label in unique_labels:
    color_3d = plt.cm.viridis(float(label) / len(unique_labels))
    label_points_3d = (labels == label)
    p_3d = data_DBSCAN[label_points_3d]
    ax.scatter(p_3d[:, 1], p_3d[:, 2], p_3d[:, 0], color=color_3d, s=30, alpha=0.7)

plt.title('3D DBSCAN Clustering')
plt.savefig('3D_DBSCAN.pdf')
plt.show()
