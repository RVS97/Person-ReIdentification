import DataClass
import NearestNeighbors
import pickle
from numpy import mean, matmul, zeros, cov, pad
from numpy.linalg import inv
from scipy.linalg import cholesky
from matplotlib import pyplot, ticker
import matplotlib.image as mpimg

## Import Data ************************************************************************************
# # Load data object
# data = DataClass.DataImport('.\\PR_data\\', 'cuhk03_new_protocol_config_labeled.mat', 'feature_data.json')

# # Save data object to pickle for faster execution
# with open('./Generated Data/data.pkl', 'wb') as f:
#     pickle.dump(data, f)

# Alternative - load data from pickle
with open('./Generated Data/data.pkl', 'rb') as f:
    data = pickle.load(f)

# ## Compute Baseline *******************************************************************************
# # Nearest neighbours with Euclidean distance
# NN_euclidean = NearestNeighbors.kNN(data, kNeighbors=5328, n_jobs=-1)
# NN_euclidean.fit('test')
# euclidean = NN_euclidean.calcScore(rank=150)
# # Uncomment the following to get the mAP
# # mAP_euclidean = NN_euclidean.calcMAP()

# # Save euclidean baseline performance to pickle for later use
# with open('./Generated Data/NNbaselineAcc.pkl', 'wb') as f:
#     pickle.dump(euclidean, f)

# # Nearest neighbours with Manhattan distance
# NN_manhattan = NearestNeighbors.kNN(data, kNeighbors=5328, p=1, n_jobs=-1)
# NN_manhattan.fit('test')
# manhattan = NN_manhattan.calcScore(rank=150)
# # Uncomment the following to get the mAP
# # mAP_manhattan = NN_manhattan.calcMAP()

# # Nearest neighbours with Cosine distance
# NN_cosine = NearestNeighbors.kNN(data, kNeighbors=5328, distMetric='cosine', n_jobs=-1)
# NN_cosine.fit('test')
# cosine = NN_cosine.calcScore(rank=150)
# # Uncomment the following to get the mAP
# # mAP_cosine = NN_cosine.calcMAP()

# ## Plot results ***********************************************************************************
# pyplot.rcParams["font.family"] = 'Times New Roman'
# fig, ax = pyplot.subplots()
# pyplot.plot(euclidean)
# pyplot.plot(manhattan)
# pyplot.plot(cosine)
# pyplot.legend(['Euclidean', 'Manhattan', 'Cosine', 'Mahalanobis'])
# pyplot.title("kNN Rank-k Accuracy", fontsize=20)
# pyplot.legend(['Euclidean', 'Manhattan', 'Cosine', 'Mahalanobis'], fontsize=16, loc=0)
# ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
# pyplot.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize=16)
# pyplot.xticks([0] + [9] + list(range(49, 160, 50)), [1] + [10] + list(range(50, 160, 50)),fontsize=16)
# pyplot.ylabel("Accuracy", fontsize=18)
# pyplot.xlabel("Rank-k", fontsize=18)
# pyplot.grid()
# pyplot.show()

# # Uncoment to print mAPs
# # print(mAP_euclidean)
# # print(mAP_manhattan)
# # print(mAP_cosine)

## Matches figure
with open('baselineKrank.pkl', 'rb') as f:
    krank = pickle.load(f)

pyplot.rcParams["font.family"] = 'Times New Roman'
fig, axes = pyplot.subplots(nrows=5, ncols=11)
pyplot.suptitle("Rank-10 lists for 5 random query images", fontsize=20)
idx = [28, 42, 77, 3, 4]
for i in range(len(idx)):
   
    imgOrg = mpimg.imread('./PR_data/images_cuhk03/'+data.getImageFileName(idx[i], 'query')[0])

    img = zeros((len(imgOrg)+20, len(imgOrg[0])+20, 3))

    img[:,:,0] = pad(imgOrg[:,:,0], ((10,10), (10,10)), 'constant', constant_values=(0,0))
    img[:,:,1] = pad(imgOrg[:,:,1], ((10,10), (10,10)), 'constant', constant_values=(0,0))
    img[:,:,2] = pad(imgOrg[:,:,2], ((10,10), (10,10)), 'constant', constant_values=(0,0))

    axes[i, 0].imshow(img)
    axes[i, 0].axis('off')
    label = data.labelsQuery[idx[i]]
    for j in range(10):
        if data.labelsGallery[krank[idx[i]][j]] == label:
            match = True
        else:
            match = False
        imgOrg = mpimg.imread('./PR_data/images_cuhk03/'+data.getImageFileName(krank[idx[i]][j], 'gallery')[0])
        img = zeros((len(imgOrg)+20, len(imgOrg[0])+20, 3))
        if not match:
            img[:,:,0] = pad(imgOrg[:,:,0], ((10,10), (10,10)), 'constant', constant_values=(1,1))
            img[:,:,1] = pad(imgOrg[:,:,1], ((10,10), (10,10)), 'constant', constant_values=(0,0))
        else:
            img[:,:,0] = pad(imgOrg[:,:,0], ((10,10), (10,10)), 'constant', constant_values=(0,0))
            img[:,:,1] = pad(imgOrg[:,:,1], ((10,10), (10,10)), 'constant', constant_values=(1,1))
        img[:,:,2] = pad(imgOrg[:,:,2], ((10,10), (10,10)), 'constant', constant_values=(0,0))

        
        axes[i, j+1].imshow(img)
        axes[i, j+1].axis('off')
pyplot.show()
exit