import DataClass
import pickle
import Optimiser
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

## Run optimiser
# trainData = data.getData('train')
# opt = Optimiser.OptimiseA(trainData[0], trainData[1], useTripplets=False)
# A = opt.optimise(learnRate=1e-3, nEpochs=300, batchSize=50, checkCost=10, logOutput=True, savePeriod=10)

## Compute Nearest Neighbors on Mahalanobis Distance **********************************************
NN = NearestNeighbors.kNN(data, kNeighbors=-1, n_jobs=-1)
NN.setTransform('./Generated Data/Opt1.pkl', isPickle=True)
NN.fit('test', applyTransform=True)
#print('Opt1 :' + str(NN.calcMAP()))
score1 = NN.calcScore(60)
#print(str(score1[0]) + " " + str(score1[9]))

NN.setTransform('./Generated Data/Opt2.pkl', isPickle=True)
NN.fit('test', applyTransform=True)
#print('Opt2 :' + str(NN.calcMAP()))
score2 = NN.calcScore(60)
#print(str(score2[0]) + " " + str(score2[9]))

with open('./Generated Data/LMNN_3_0.0001.pkl', 'rb') as f:
    lmnn = pickle.load(f)
NN.setTransform(lmnn.metric())
NN.fit('test', applyTransform=True)
scoreLMNN = NN.calcScore(60)
#print('LMNN :' + str(NN.calcMAP()))
#print(str(scoreLMNN[0]) + " " + str(scoreLMNN[9]))
#mAP_mahalanobis = NN_mahalanobis.calcMAP()
#print(mAP_mahalanobis)

## Load baseline and plot comparison **************************************************************
with open('./Generated Data/basekNNRanks.pkl', 'rb') as f:
    baseline = pickle.load(f)

pyplot.rcParams["font.family"] = 'Times New Roman'
fig, ax = pyplot.subplots(figsize=(6, 4), dpi=100)
pyplot.plot(baseline, linewidth=1)
pyplot.plot(scoreLMNN, linewidth=1)
pyplot.plot(score1, linewidth=1)
pyplot.plot(score2, linewidth=1)
pyplot.legend(['Baseline', 'LMNN', 'Proposed 1', 'Proposed 2'], fontsize=16, loc=0)
pyplot.title("LMNN and Proposed Rank-k Accuracy", fontsize=22)
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