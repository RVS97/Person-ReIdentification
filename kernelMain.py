import DataClass
import pickle
import NearestNeighbors
from numpy import zeros, argmax, where, argmin, logical_not
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

# rbf_kernel = NearestNeighbors.kernel('rbf', gamma=-0.1)
# AChi2_kernel = NearestNeighbors.kernel('AChi2', gamma = 0.1)
# Chi2_kernel = NearestNeighbors.kernel('Chi2', gamma = -0.1)
# laplacian_kernel = NearestNeighbors.kernel('laplacian', gamma = -0.1)

# NN_Kernel = NearestNeighbors.kNN(data, kNeighbors=-1, n_jobs=-1)
# NN_Kernel.kernelFit('test', rbf_kernel)
# rbf = NN_Kernel.calcScore(150)
# NN_Kernel.kernelFit('test', Chi2_kernel)
# Chi2 = NN_Kernel.calcScore(150)
# NN_Kernel.kernelFit('test', laplacian_kernel)
# laplacian = NN_Kernel.calcScore(150)

# with open('./Generated Data/rbfScore.pkl', 'rb') as f:
#     rbf = pickle.load(f)

# with open('./Generated Data/Chi2Score.pkl', 'rb') as f:
#     Chi2 = pickle.load(f)

# with open('./Generated Data/laplacianScore.pkl', 'rb') as f:
#     laplacian = pickle.load(f)

# ## Load baseline and plot comparison **************************************************************
# with open('./Generated Data/NNbaselineAcc.pkl', 'rb') as f:
#     baseline = pickle.load(f)

lowNN = NearestNeighbors.kNN(data, kNeighbors=-1, n_jobs=-1)
lowNN.setTransform('./Generated Data/firstOpt/Epoch 11.pkl', isPickle=True)
lowNN.fit('test', applyTransform=True)
score1 = lowNN.calcScore(60)

highNN = NearestNeighbors.kNN(data, kNeighbors=-1, n_jobs=-1)
highNN.setTransform('./Generated Data/firstOpt/Epoch 300.pkl', isPickle=True)
highNN.fit('test', applyTransform=True)
score2 = highNN.calcScore(60)

pyplot.rcParams["font.family"] = 'Times New Roman'
fig, ax = pyplot.subplots(figsize=(6, 4), dpi=100)
# pyplot.plot(baseline[0:60], linewidth=1)
# pyplot.plot(rbf[0:60], linewidth=1)
# pyplot.plot(Chi2[0:60], linewidth=1)
# pyplot.plot(laplacian[0:60], linewidth=1)
pyplot.plot(score1[0:60], linewidth=1)
pyplot.plot(score2[0:60], linewidth=1)
# pyplot.legend(['Baseline', 'Gaussian Kernel', 'Chi Squared Kernel', 'Laplacian Kernel'], fontsize=16, loc=0)
pyplot.legend(['10 Epochs', '300 Epochs'], fontsize=16, loc=0)
pyplot.title("Rank-k Accuracy", fontsize=22)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
pyplot.ylim(bottom=0.46)
pyplot.yticks([0.5, 0.75, 1], fontsize=16)
pyplot.xticks([0] + list(range(9, 69, 10)), [1] + list(range(10, 70, 10)),fontsize=16)
pyplot.ylabel("Accuracy", labelpad=-5, fontsize=18)
pyplot.xlabel("Rank-k", labelpad=-3, fontsize=18)
pyplot.grid()
pyplot.subplots_adjust(top=0.905, bottom=0.145, right=0.97, left=0.145, hspace=0.2)
pyplot.show()
exit
exit