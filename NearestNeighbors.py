from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, precision_score, label_ranking_average_precision_score, accuracy_score
from sklearn.metrics.pairwise import rbf_kernel, additive_chi2_kernel, laplacian_kernel, sigmoid_kernel, chi2_kernel
from numpy import array, where, ones, append, logical_and, logical_not, sum, matmul, zeros, mean, multiply, tril, argsort, around
import pickle
from scipy.linalg import cholesky
from matplotlib import pyplot

class kNN:
    def __init__(self, dataObj, kNeighbors, distMetric='minkowski', p=2, metric_params=None, n_jobs=None):
        # Data object with training/testdata
        self.data = dataObj

        if kNeighbors == -1:
            self.useAllNeighbors = True
        else:
            self.useAllNeighbors = False
            kNeighbors = kNeighbors

        # Create kNN classifier
        self.nn = NearestNeighbors(n_neighbors=kNeighbors, metric=distMetric, p=p, metric_params=metric_params, n_jobs=n_jobs)

    def fit(self, featType, nSplits=3, randState=1010101, applyTransform=False, customData = []):
        if featType == 'train':
            if customData == []:
                train, validation = self.data.getData('train')
                trainData = train[0]
            else:
                trainData = customData
            self.nn.fit(trainData)
            self.krank = self.nn.kneighbors(trainData)[1]
            return

        elif featType == 'test':
            query, gallery = self.data.getData('test')
        elif featType == 'validation':
            query, gallery = self.data.getData(featType)

        if self.useAllNeighbors:
            self.nn.set_params(n_neighbors=len(gallery[0]))

        if applyTransform:
            query[0] = matmul(query[0], self.U.T)
            gallery[0] = matmul(gallery[0], self.U.T)
            
        self.nn.fit(gallery[0])
        self.krank = self.nn.kneighbors(query[0])[1]
        # for query, gallery in self.data.getData('test'):
        #     self.nn.fit(gallery[0])
        #     # Save indices of k nearest neighbors
        #     self.krank.append(self.nn.kneighbors(array(query[0]).reshape(1, -1))[1])
        k = len(self.krank[0])
        for i in range(len(self.krank)):
            qLab = query[1][i]
            qCam = query[2][i]
            labs = gallery[1][self.krank[i]]
            cams = gallery[2][self.krank[i]]
            correctPos = where(logical_not(logical_and(labs==qLab, cams==qCam)))
            self.krank[i] = append(self.krank[i][correctPos], array([-1]*(k-len(correctPos[0]))))
        maxEmpty = max(sum(self.krank==-1, axis=1))
        self.krank = self.krank[:,0:k-maxEmpty]

    # def fitModel(self, features, labels):
    #     # Fit model. Features consists of rows of features
    #     self.nn.fit(features, labels)

    def modParams(self, kNeighbors, distMetric, p='minkowski', metric_params=2, n_jobs=None):
        self.nn.set_params(n_neighbors=kNeighbors, metric=distMetric, p=p, metric_params=metric_params, n_jobs=n_jobs)

    def setTransform(self, A, isPickle=False):
        if isPickle:
            with open(A, 'rb') as f:
                A = pickle.load(f)
        
        self.U = cholesky(A, lower=False)
    
    def setTransMat(self, A, isPickle=False):
        if isPickle:
            with open(A, 'rb') as f:
                A = pickle.load(f)
        self.transMat = A

    def calcScore(self, rank, plot=False):
        self.rankAccs = zeros(rank)

        for i in range(len(self.krank)):
            for j in range(rank):
                matches = self.data.labelsGallery[self.krank[i][0:j+1]]==self.data.labelsQuery[i]
                positiveMatches = sum(matches)
                if positiveMatches > 0:
                    self.rankAccs[j:rank] += 1
                    break

        self.rankAccs = self.rankAccs/len(self.krank)
        
        if plot:
            self.plotAccs(self.rankAccs)

        return self.rankAccs

    def calcMAP(self):
        self.rankMAp = 0
        for i in range(len(self.krank)):
            self.rankMAp += self.calcAP(self.data.labelsQuery[i], self.data.labelsGallery[self.krank[i]])
        self.rankMAp = self.rankMAp/len(self.krank)
        return self.rankMAp

    def plotAccs(self, rank):
        fig, axs = pyplot.subplots()
        axs.plot(rank)
        pyplot.show()
    
    def calcAP(self, trueLabel, neighbors):
        nNeighbors = len(neighbors)
        nMatches = sum(trueLabel==neighbors)

        precision = zeros(nNeighbors)
        recall = zeros(nNeighbors)

        if nMatches == 0:
            return 0

        recallInc = 1/nMatches

        #trueLabelArr = array([trueLabel]*nNeighbors)

        nPoints = 11
        interp = zeros(nPoints)

        for i in range(len(precision)):
            precision[i] = mean(trueLabel==neighbors[0:i+1])

            if i == 0:
                recall[i] = recallInc*(trueLabel == neighbors[i])
            else:
                recall[i] = recall[i-1]
                if trueLabel == neighbors[i]:
                    recall[i] = recall[i] + recallInc
        
        #recall = matmul((trueLabel==neighbors)*recallInc, tril(ones((len(neighbors), len(neighbors)))))

        recall = around(recall, 10)
        for i in range(nPoints):
            idx = min(where(i*0.1<=recall)[0])
           
            interp[i] = max(precision[idx:len(precision)])

        return mean(interp)

    def kernelFit(self, featType, kernel, applyTransform=False):
        self.krank = []
        if featType == 'train':
                pass
        elif featType == 'test':
            query, gallery = self.data.getData('test')
            if applyTransform:
                    query[0] = matmul(query[0], self.U.T)
                    gallery[0] = matmul(gallery[0], self.U.T)
            for i in range(len(query[0])):

                dist = kernel.transform(query[0][i].reshape(1, -1), gallery[0])
                idx = argsort(dist, axis=0)
                #idx = argsort(dist, axis=0)[::-1]

                qLab = query[1][i]
                qCam = query[2][i]
                labs = gallery[1][idx]
                cams = gallery[2][idx]
                correctPos = where(logical_not(logical_and(labs==qLab, cams==qCam)).flatten())
                self.krank.append(append(idx[correctPos], array([-1]*(len(gallery[0])-len(correctPos[0])))))

            self.krank = array(self.krank)
            maxEmpty = max(sum(self.krank==-1, axis=1))
            self.krank = self.krank[:,0:len(gallery[0])-maxEmpty]

class kernel:
    def __init__(self, kernelType, **params):
        self.type = kernelType
        if self.type == 'rbf':
            self.gamma = params.get("gamma")
        elif self.type == 'Chi2':
            self.gamma = params.get("gamma")
        elif self.type == 'laplacian':
            self.gamma = params.get("gamma")
        elif self.type == 'sigmoid':
            self.gamma = params.get("gamma")
            self.coef0 = params.get("coef0")
    def transform(self, X, Y):
        if self.type == 'rbf':
            return rbf_kernel(X, Y, self.gamma)[0]
        elif self.type == 'Chi2':
            return chi2_kernel(X, Y, self.gamma)[0]
        elif self.type == 'AChi2':
            return -additive_chi2_kernel(X, Y)[0]
        elif self.type == 'laplacian':
            return laplacian_kernel(X,Y,self.gamma)[0]
        elif self.type == 'sigmoid':
            return sigmoid_kernel(X,Y,self.gamma,self.coef0)[0]