import DataClass
import NearestNeighbors
import MiscLibrary
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
import pickle

from metric_learn.lmnn import LMNN
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

MiscLibrary.plotMatPerformance(150, './Generated Data/02-15_12-05/Epoch 491.pkl', data)

exit