from numpy import zeros
import NearestNeighbors
from matplotlib import pyplot, ticker
import pickle

def getRankKAcc(k, rankLists, dataObj):
    rankAccs = zeros(k)
    
    for i in range(len(rankLists)):
        for j in range(k):
            matches = dataObj.labelsGallery[rankLists[i][0:j+1]]==dataObj.labelsQuery[i]
            positiveMatches = sum(matches)
            if positiveMatches > 0:
                rankAccs[j:k] += 1
                break

    rankAccs = rankAccs/len(rankLists)

    return rankAccs

def plotMatPerformance(k, A, dataObj, isPickle=True, compareWithBaseline=True):
    NN = NearestNeighbors.kNN(dataObj, kNeighbors=k, n_jobs=-1)
    NN.setTransform(A, isPickle=isPickle)
    NN.fit('test', applyTransform=True)

    newAcc = getRankKAcc(k, NN.krank, dataObj)

    with open('./Generated Data/NNbaselineAcc.pkl', 'rb') as f:
        baseline = pickle.load(f)

    pyplot.rcParams["font.family"] = 'Times New Roman'
    fig, ax = pyplot.subplots()
    pyplot.plot(baseline)
    pyplot.plot(newAcc)
    pyplot.legend(['Baseline', 'New'], fontsize=16, loc=0)
    pyplot.title("kNN Rank-k Accuracy", fontsize=20)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    pyplot.yticks(fontsize=16)
    pyplot.xticks([0] + [9] + list(range(49, 160, 50)), [1] + [10] + list(range(50, 160, 50)),fontsize=16)
    pyplot.ylabel("Accuracy", fontsize=18)
    pyplot.xlabel("Rank-k", fontsize=18)
    pyplot.grid()
    pyplot.show()
    exit