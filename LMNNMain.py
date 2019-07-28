import DataClass
import NearestNeighbors
import pickle
from random import choice, seed
from metric_learn.lmnn import LMNN
from numpy import mean, matmul, zeros, cov, where, unique, logical_and, logical_not
from numpy.linalg import inv
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

## LMNN *******************************************************************************************
# # Get query and gallery set
# [train, validation] = data.getData('train')

ks = [3, 5]
regs = [0.1, 0.01, 0.001, 0.0001]
# for k in ks:
#     for reg in regs:
#         lmnn = LMNN(k=k, min_iter=50, max_iter=10000, learn_rate=1e-07, regularization=reg, convergence_tol=0.001, verbose=True)
#         lmnn.fit(train[0], train[1])
#         with open('./Generated Data/LMNN_' + str(k) + '_' + str(reg) + '.pkl', 'wb') as f:
#             pickle.dump(lmnn, f)

## Compute mAPs for LMNN **************************************************************************
for k in ks:
    for reg in regs:
        NN = NearestNeighbors.kNN(data, kNeighbors=-1, n_jobs=-1)
        with open('./Generated Data/LMNN_' + str(k) + '_' + str(reg) + '.pkl', 'rb') as f:
            lmnn = pickle.load(f)
        NN.setTransform(lmnn.metric())
        NN.fit('validation', applyTransform=True)
        print('k: ' + str(k) + ', reg: ' + str(reg) + str(NN.calcMAP()))