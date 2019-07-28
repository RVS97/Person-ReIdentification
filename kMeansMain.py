import DataClass
import pickle
from numpy import mean, matmul, zeros, cov, where, logical_not, logical_and
from numpy.linalg import inv
from matplotlib import pyplot, ticker
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

## Import Data ************************************************************************************
# # Load data object
# data = DataClass.DataImport('.\\PR_data\\', 'cuhk03_new_protocol_config_labeled.mat', 'feature_data.json')

# # Save data object to pickle for faster execution
# with open('./Generated Data/data.pkl', 'wb') as f:
#     pickle.dump(data, f)

# Alternative - load data from pickle
with open('./Generated Data/data.pkl', 'rb') as f:
    data = pickle.load(f)

## Compute k-Means Baseline ***********************************************************************
# Define number of clusters to perform fit to
clustersList = [1, 2, 3, 4, 5, 10, 50, 100, 500, 700]
# Define empty list for returned accuracies
accuracy = []

# Get query and gallery set
[query, gallery] = data.getData('test')

# Loop through the different number of clusters and get accuracy
for nClusters in clustersList:
    ## Perform clustering *************************************************************************
    # Load kMeans object
    kM = KMeans(n_clusters=nClusters, random_state=1010101, n_jobs=-1)

    # Fit kMeans to the features in gallery
    kM.fit(gallery[0])

    # Create nearest neighbors object with Euclidean distance
    nn = NearestNeighbors(n_neighbors=1, metric='minkowski', p=2, n_jobs=-1)

    # Get the calculated cluster centers and the cluster that each gallery feature vector has been assigned to
    clusterCenters = kM.cluster_centers_
    galleryLabels = kM.labels_

    # Predict the clusters that the query feature vectors would be assigned to
    closestClusters = kM.predict(query[0])

    # Loop through all the query samlpes and get the rank-k accuracies
    rank_k = []
    for i in range(len(query[0])):
        # Get the cluster id assigned to the query in question
        queryLabel = closestClusters[i]
        # Get the gallery feature vectors that share the same cluster label (i.e. gallery vector that belong to the cluster)
        clusterElements = where(galleryLabels==queryLabel)[0]
        
        # Set the number of neigbhors to find to the total number of members in the cluster
        nn.set_params(n_neighbors=len(clusterElements))
        # Fit the nearest neighbors algorithm with the elements of the cluster
        nn.fit(gallery[0][clusterElements])
        # Find the nearest neigbhors of the query feature  vector
        kneighbors = nn.kneighbors(query[0][i].reshape(1, -1), return_distance=False)[0]

        # Remove the neighbors that share the same label and cam id
        qLab = query[1][i]                  # Get the query label
        qCam = query[2][i]                  # Get the query cam id
        labs = gallery[1][kneighbors]       # Get all the labels from the neighbors
        cams = gallery[2][kneighbors]       # Get all the cam ids from the nieghbors
        # Get the indeces that do not have the same label and cam id
        correctPos = where(logical_not(logical_and(labs==qLab, cams==qCam)).flatten())
        # Append the ranklist to rank_k
        rank_k.append(kneighbors[correctPos])

    ## Get rank-k accuracy ************************************************************************
    k = 150

    acc = zeros(k)
    for j in range(k):
        for i in range(len(rank_k)):
            # If there is one or more matches, increment acc counter at position j
            if sum(data.labelsGallery[rank_k[i][0:j+1]]==data.labelsQuery[i]) > 0:
                acc[j] += 1

    # Get percentage of matches
    acc = acc/len(rank_k)

    # Append to 'accuracy'
    accuracy.append(acc)

    # Save current accuracy variable
    with open('./Generated Data/clustersResultsAll.pkl', 'wb') as f:
        pickle.dump(accuracy, f)

## Plot results ***********************************************************************************
# # Open saved accuracies
# with open('./Generated Data/clustersResultsAll.pkl', 'rb') as f:
#     accuracy = pickle.load(f)

pyplot.rcParams["font.family"] = 'Times New Roman'
fig, ax = pyplot.subplots()
for i in range(len(clustersList)):
    pyplot.plot(accuracy[i]) 

pyplot.legend(["1", "2", "3", "4", "5", "10", "50", "100", "500", "700"], title="Number of clusters", fontsize=12, title_fontsize=12, loc=5, bbox_to_anchor=(1.35,0.5))
pyplot.title("k-Means Rank-k Accuracy for varying number of clusters", fontsize=18)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
pyplot.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=16)
pyplot.xticks([0] + [9] + list(range(49, 160, 50)), [1] + [10] + list(range(50, 160, 50)), fontsize=16)
pyplot.ylabel("Accuracy", fontsize=18)
pyplot.xlabel("Rank-k", fontsize=18)
pyplot.grid()
pyplot.show()

exit