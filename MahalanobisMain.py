import DataClass
import NearestNeighbors
import pickle

from numpy import mean, matmul, zeros, cov
from numpy.linalg import inv
from scipy.linalg import cholesky
from matplotlib import pyplot, ticker

## Import Data ************************************************************************************
# # Load data object
# data = DataClass.DataImport('.\\PR_data\\', 'cuhk03_new_protocol_config_labeled.mat', 'feature_data.json')

# # Save data object to pickle for faster execution
# with open('./Generated Data/data.pkl', 'wb') as f:
#     pickle.dump(data, f)

# Alternative - load data from pickle
with open('./Generated Data/data.pkl', 'rb') as f:
    data = pickle.load(f)

## Compute Covariance Matrix **********************************************************************
# Get query and gallery set
[train, validation] = data.getData('train')

# Compute covariance matrix of the training data and its inverse
covMat = cov(train[0].T)
mahMat = inv(covMat)

# # Save Inverse covariance matrix
# with open('./Generated Data/mahMat.pkl', 'wb') as f:
#     pickle.dump(mahMat, f)

## Compute Nearest Neighbors on Mahalanobis Distance **********************************************
# NN_mahalanobis = NearestNeighbors.kNN(data, kNeighbors=-1, n_jobs=-1)
# NN_mahalanobis.setTransform('./Generated Data/mahMat.pkl', isPickle=True)
# NN_mahalanobis.fit('test', applyTransform=True)
# mahalanobisScore = NN_mahalanobis.calcScore(150)
#mAP_mahalanobis = NN_mahalanobis.calcMAP()
#print(mAP_mahalanobis)
# with open('./Generated Data/MahalanobisScore.pkl', 'wb') as f:
#     pickle.dump(mahalanobisScore, f)

## Load baseline and plot comparison **************************************************************
with open('./Generated Data/NNbaselineAcc.pkl', 'rb') as f:
    baseline = pickle.load(f)

with open('./Generated Data/MahalanobisScore.pkl', 'rb') as f:
    mahalanobisScore = pickle.load(f)

pyplot.rcParams["font.family"] = 'Times New Roman'
fig, ax = pyplot.subplots(figsize=(6, 4), dpi=100)
pyplot.plot(baseline)
pyplot.plot(mahalanobisScore)
pyplot.legend(['Baseline', 'Mahalanobis'], fontsize=16, loc=0)
pyplot.title("kNN Rank-k Accuracy", fontsize=22)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
pyplot.yticks([0.25, 0.5, 0.75, 1], fontsize=16)
pyplot.xticks([0] + [9] + list(range(49, 160, 50)), [1] + [10] + list(range(50, 160, 50)),fontsize=16)
pyplot.ylabel("Accuracy", labelpad=-5, fontsize=18)
pyplot.xlabel("Rank-k", labelpad=-3, fontsize=18)
pyplot.grid()
pyplot.subplots_adjust(top=0.905, bottom=0.145, right=0.97, left=0.145, hspace=0.2)
pyplot.show()
exit