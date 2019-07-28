from scipy.io import loadmat
from numpy import array, unique, where, logical_or, logical_not, logical_and
import json
import random
from math import floor
from random import choice, seed

class DataImport:
    def __init__(self, filePath, matFileName, jsonFileName, randState=1010101, valSetSize=0.13):
        data = loadmat(filePath + matFileName)

        # Images corresponding to feature vectors
        self.filelist = data["filelist"].flatten()
        # All labels
        labels = data["labels"].flatten()
        # Cam ids for feature vectors
        camId = data["camId"].flatten()
        # Indeces of the feature vectors to be used for training
        self.train_idx = data["train_idx"].flatten()-1
        # Gallery indeces for test set
        self.gallery_idx = data["gallery_idx"].flatten()-1
        # Query indeces for test set
        self.query_idx = data["query_idx"].flatten()-1

        # Feature vectors
        with open(filePath + jsonFileName, 'r') as f:
            features = array(json.load(f))

        # Create training sets
        self.labelsTrain = labels[self.train_idx]
        self.camIdTrain = camId[self.train_idx]
        self.featuresTrain = features[self.train_idx]

        self.createValidationSet(valSetSize, randState)

        self.splitValidationSet(randState)

        # Create gallery set (for test set)
        self.labelsGallery = labels[self.gallery_idx]
        self.camIdGallery = camId[self.gallery_idx]
        self.featuresGallery = features[self.gallery_idx]

        # Create query set (for test set)
        self.labelsQuery = labels[self.query_idx]
        self.camIdQuery = camId[self.query_idx]
        self.featuresQuery = features[self.query_idx]
    
    def createValidationSet(self, valSetSize, randState):
        random.seed(randState)
        labels = unique(self.labelsTrain)
        valLabels = array(sorted(random.sample(range(0, len(labels)), floor(len(labels)*valSetSize))))

        valIdx = self.labelsTrain == valLabels[0]
        for i in range(1, len(valLabels)):
            valIdx = logical_or(valIdx, self.labelsTrain == valLabels[i])
        
        self.valIdx = where(valIdx)
        self.trainIdx = where(logical_not(valIdx))

    def splitValidationSet(self, randState):
        valLabs = self.labelsTrain[self.valIdx]
        valCams = self.camIdTrain[self.valIdx]

        # Split validation set into gallery and query
        galleryIdx = []
        queryIdx = []

        seed(1010101)

        for label in unique(valLabs):
            matches = valLabs==label
            # Get query for cam 1
            cam1 = where(logical_and(valCams==1, matches))[0]
            # Get a random element
            qIdx = choice(cam1)
            queryIdx.append(qIdx)
            galleryIdx.extend(cam1[where(logical_not(cam1==qIdx))])

            # Get query for cam 2
            cam2 = where(logical_and(valCams==2, matches))[0]
            # Get a random element
            qIdx = choice(cam2)
            queryIdx.append(qIdx)
            galleryIdx.extend(cam2[where(logical_not(cam2==qIdx))])

        self.validationQueryIdx = queryIdx
        self.validationGalleryIdx = galleryIdx


    def getData(self, featType):
        if featType == 'train':
            train = self.featuresTrain[self.trainIdx], self.labelsTrain[self.trainIdx], self.camIdTrain[self.trainIdx]
            val = self.featuresTrain[self.valIdx], self.labelsTrain[self.valIdx], self.camIdTrain[self.valIdx]
            return [train, val]

        elif featType == 'validation':
            valQuery = [self.featuresTrain[self.valIdx][self.validationQueryIdx], self.labelsTrain[self.valIdx][self.validationQueryIdx], self.camIdTrain[self.valIdx][self.validationQueryIdx]]
            valGallery = [self.featuresTrain[self.valIdx][self.validationGalleryIdx], self.labelsTrain[self.valIdx][self.validationGalleryIdx], self.camIdTrain[self.valIdx][self.validationGalleryIdx]]
            return valQuery, valGallery
            
        elif featType == 'test':
            return [self.featuresQuery, self.labelsQuery, self.camIdQuery], [self.featuresGallery, self.labelsGallery, self.camIdGallery]
            # for i in range(len(self.featuresQuery)):
            #     featQ = self.featuresQuery[i]
            #     labQ = self.labelsQuery[i]
            #     camIdQ = self.camIdQuery[i]
            #     galleryIdx = where(logical_not(logical_and(self.camIdGallery==camIdQ, self.labelsGallery==labQ)))
            #     featGallery = self.featuresGallery[galleryIdx]
            #     labGallery = self.labelsGallery[galleryIdx]
            #     yield [[featQ, labQ], [featGallery, labGallery]]

    def getImageFileName(self, idx, dataset):
        if dataset=='query':
            return self.filelist[self.query_idx[idx]]
        elif dataset=='gallery':
            return self.filelist[self.gallery_idx[idx]]