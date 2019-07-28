from numpy import eye, matmul, zeros, tile, multiply, ones, where, logical_not, log, exp, append
from random import sample, seed, choice
import datetime
import os
import pickle
import NearestNeighbors
from math import floor
from sklearn.datasets import make_spd_matrix

# Optimisation class
class OptimiseA:
    def __init__(self, train, val, useTripplets=False):
        # Store training and validation sets in class
        self.featTrain = train[0]
        self.labTrain = train[1]
        self.featVal = val[0]
        self.labVal = val[1]
        self.useTripplets = useTripplets

    # Optimise function
    def optimise(self, learnRate, nEpochs, batchSize, checkCost, logOutput=True, savePeriod=10, randState=1010101):
        seed(randState)

        # Create log file
        if logOutput:
            # Genearte folder name
            now = datetime.datetime.now()
            folderName = now.strftime("%H-%M %d-%m")

            # Create folder
            if not os.path.exists("./" + folderName):
                os.makedirs("./" + folderName)
                logDir = "./" + folderName + "/Config.txt"

                # Save parameters to log
                with open(logDir, 'w') as f_log:
                    f_log.write(str(now)+"\n")
                    f_log.write("Learn rate: " + str(learnRate) + "\n")
                    f_log.write("Number of epochs: " + str(nEpochs) + "\n")
                    f_log.write("Batch size: " + str(batchSize) + "\n")
                    f_log.write("Check cost every " + str(checkCost) + " epochs.\n")
                    f_log.write("Random seed: " + str(randState) + "\n")

        # Define initial definition of A
        A = eye(len(self.featTrain[0]))

        # Set previous cost to infinity
        self.prevCost = float('inf')
        
        idx = sample(range(len(self.featTrain)), 100)
        idxVal = sample(range(len(self.featVal)), 100)


        # Validation points
        P_hard_val = self.getPointSet([self.featVal, self.labVal])
        # Random points
        P_soft_val = self.getPointSet([self.featVal, self.labVal], pickWorstCase=False)
        # Mix
        idx50_hard = sample(range(len(self.featVal)), floor(len(P_hard_val)/2))
        idx50_soft = sample(range(len(self.featVal)), floor(len(P_hard_val)/2))
        # P_val = append(P_hard_val[idx50_hard], P_soft_val[idx50_soft], axis=0)

        valPlot = []
        valPlotidx = 0

        # Run epochs
        for t in range(nEpochs):
            # Update cost every 'checkCost' epochs
            if t%checkCost==0:
                if self.useTripplets:
                    # Update point set P
                    # Get point set (triple) of (query, farthest similar, closest disimilar)
                    # Hard points
                    P_hard = self.getPointSet([self.featTrain, self.labTrain])
                    # Random points
                    P_soft = self.getPointSet([self.featTrain, self.labTrain], pickWorstCase=False)
                    # Mix
                    idx50_hard = sample(range(len(self.featTrain)), floor(len(P_hard)/2))
                    idx50_soft = sample(range(len(self.featTrain)), floor(len(P_hard)/2))
                    P = append(P_hard[idx50_hard], P_soft[idx50_soft], axis=0)


                    # Training set ************************************************************
                    # Get current cost
                    #self.currentCost = self.getTotalCost(A, [self.featTrain[idx], self.labTrain[idx]])
                    self.trainCost = self.calcCost(A, P_hard, 1) + self.calcCost(A, P_soft, 1) 

                    # Validation set *********************************************************
                    # Get validation cost
                    validationCost = self.calcCost(A, P_hard_val, 1) + self.calcCost(A, P_soft_val, 1) 
                else:
                    self.trainCost = self.getTotalCost(A, [self.featTrain[idx], self.labTrain[idx]])
                    validationCost = self.getTotalCost(A, [self.featVal[idxVal], self.labVal[idxVal]])

                # If the previous cost was smaller, half the learning rate, reset A_new to A and re-run, otherwise update A and 'prevCost' variables
                if self.prevCost < validationCost:
                    learnRate = learnRate/2
                    #A_new = A
                else:
                    #A = A_new
                    self.prevCost = validationCost

                

                # Write log output
                if logOutput:
                    with open(logDir, 'a') as f_log:
                        now = datetime.datetime.now()
                        timeStamp = now.strftime("%H:%M")
                        f_log.write("Epoch " + str(t+1) + " of " + str(nEpochs) + " - " + str(timeStamp) + "\n")
                        f_log.write("Current cost: " + str(self.trainCost) + ", learn rate: " + str(learnRate) + "\n")
                        f_log.write("Validation cost: " + str(validationCost) + "\n")
                print(self.trainCost, validationCost)

                valPlot.append(validationCost)
                valPlotidx = valPlotidx + 1

                
                
                # NN = NearestNeighbors.kNN(data, kNeighbors=20, n_jobs=-1)
                # NN.setTransform(A)
                # NN.fit('validation', applyTransform=True)

                
                

            # Initialise gradient variable
            grad = zeros(A.shape)

            # Get a random batch of points to optimise A on, and perform optimisation
            if self.useTripplets:
                batchIdx = sample(range(len(P)), batchSize)
            else:
                batchIdx = sample(range(len(self.featTrain)), batchSize*2)
            for iBatch in range(batchSize):
                if self.useTripplets:
                    ref = P[batchIdx[iBatch], 0]        # Anchor point
                    sim = P[batchIdx[iBatch], 1]        # Similar point to bring close
                    diss = P[batchIdx[iBatch], 2]       # Disimilar point to bring apart

                    # Get gradient between reference and similar point and add to 'grad'
                    y = self.featTrain[ref]-self.featTrain[sim]
                    y = y.reshape((1, len(y)))
                    grad += matmul(y.T, y)

                    # Get gradient between reference and dissimilar point and subtract from 'grad'
                    y = self.featTrain[ref]-self.featTrain[diss]
                    y = y.reshape((1, len(y)))
                    grad -= matmul(y.T, y)
                else:
                    i = batchIdx[iBatch*2]
                    j = batchIdx[iBatch*2+1]
                    l = self.compare(self.labTrain[i], self.labTrain[j])
                    y = self.featTrain[i]-self.featTrain[j]
                    y = y.reshape((1, len(y)))
                    grad += l*matmul(y.T, y)

            # Update A_new with average gradient
            grad = grad/(batchSize*2)
            #A_new = A_new-learnRate*grad
            A = A-learnRate*grad

            # Save A to file every 'savePeriod'
            if t%savePeriod==0 and logOutput:
                fileName = "Epoch " + str(t+1) + ".pkl"
                f = open("./" + folderName + "/" + fileName, 'wb')
                pickle.dump(A, f)
                f.close()

        # Save end A to file
        if logOutput:
                fileName = "Epoch " + str(t+1) + ".pkl"
                f = open("./" + folderName + "/" + fileName, 'wb')
                pickle.dump(A, f)
                f.close()

        # Return A
        return A


    def calcCost(self, A, pointsSet, margin):
        Anchors = pointsSet[:, 0]
        Similar = pointsSet[:, 1]
        Dissimilar = pointsSet[:, 2]

        ysim = self.featTrain[Anchors] - self.featTrain[Similar]
        ydis = self.featTrain[Anchors] - self.featTrain[Dissimilar]

        dsim = matmul(multiply(matmul(ysim, A), ysim), ones((len(ysim[0]), 1)))
        ddis = matmul(multiply(matmul(ydis, A), ydis), ones((len(ysim[0]), 1)))

        #loss = amax(margin*ones((len(ysim), 1))+dsim-ddis, axis=0)

        # loss = log(1+exp(margin*ones((len(ysim), 1))+dsim-ddis))
        loss = margin*ones((len(ysim), 1))+dsim-ddis
        loss = loss.clip(min=0)

        return sum(loss)
    
    def getTotalCost(self, A, pointsSet):
        feat = pointsSet[0]
        labs = pointsSet[1]
        C = 0
        n = len(feat)
        y = zeros((int(n*(n-1)/2),len(feat[0])))
        l = zeros(int(n*(n-1)/2))
        idx = 0
        for i in range(len(feat)):
            for j in range(i+1, len(feat)):
                y[idx,:] = feat[i]-feat[j]
                l[idx] = self.compare(labs[i], labs[j])
                idx = idx + 1
        
        C = matmul(l, matmul(multiply(matmul(y, A), y), ones((len(feat[0]), 1))))
        return C

    def compare(self, labA, labB):
        if labA == labB:
            return 1
        else:
            return -1

    def getPointSet(self, data, pickWorstCase=True):

        NN = NearestNeighbors.kNN(None, kNeighbors=len(data[0]), n_jobs=-1)
        NN.fit('train', customData = data[0])

        P = zeros((len(NN.krank),3), dtype=int)
        P[:, 0] = range(len(data[0]))

        for i in range(len(NN.krank)):
            labS = data[1][i]
            mask = data[1][NN.krank[i]]==labS
            if pickWorstCase:
                P[i, 1] = NN.krank[i][max(where(mask)[0])]
                P[i, 2] = NN.krank[i][choice(where(logical_not(mask))[0])]
            else:
                P[i, 1] = NN.krank[i][choice(where(mask)[0])]
                P[i, 2] = NN.krank[i][choice(where(logical_not(mask))[0])]

        return P