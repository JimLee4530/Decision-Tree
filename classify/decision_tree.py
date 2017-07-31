import numpy as np
import pandas as pd
from math import log

import operator

class ID3(object):
    # implement the DecisionTree on your own.
    def __init__(self, X, labels):
        self.X = X
        self.labels = labels

    def calcShannonEnt(self, dataSet):
        numEntries = len(dataSet)
        labelCounts = {}
        label_tmp = dataSet[:, -1].astype(int)
        labelCounts_tmp = np.bincount(label_tmp)
        for label, num in enumerate(labelCounts_tmp):
            labelCounts[label] = num
        shannonEnt = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key])/numEntries + 1e-15
            # print prob
            shannonEnt += - prob * log(prob, 2)
        return shannonEnt

    def GiniD(self, dataSet):
        numEntries = len(dataSet)
        labelCounts = {}
        label_tmp = dataSet[:, -1].astype(int)
        labelCounts_tmp = np.bincount(label_tmp)
        for label, num in enumerate(labelCounts_tmp):
            labelCounts[label] = num
        p = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key]) / numEntries
            # print prob
            p += prob**2
        return 1.0 - p

    def splitDataSet(self, dataSet, axis, value):
        retDataSet = np.concatenate((dataSet[dataSet[:,axis] == value, :axis], dataSet[dataSet[:,axis] == value, axis+1:]), axis=1)
        return np.array(retDataSet)

    def chooseBestFeatureToSplit(self, dataSet):
        numFeatures = len(dataSet[0]) - 2
        baseEntropy = self.calcShannonEnt(dataSet)
        bestInfoGain = 0.0
        bestFeature = -1
        for i in range(numFeatures):
            featList = dataSet[:, i+1].ravel()
            uniqueVals = set(featList)
            newEntopy = 0.0
            for value in uniqueVals:
                subDataSet = self.splitDataSet(dataSet, i+1, value)
                prob = len(subDataSet)/ float(len(dataSet))
                newEntopy += prob * self.calcShannonEnt(subDataSet)
            infoGain = baseEntropy - newEntopy
            if (infoGain > bestInfoGain):
                bestInfoGain = infoGain
                bestFeature = i+1
        return bestFeature

    def chooseBestFeatureToSplitByGiniIndex(self, dataSet):
        numFeatures = len(dataSet[0]) - 2
        minGiniIndex = self.GiniD(dataSet)
        bestFeature = -1
        for i in range(numFeatures):
            featList = dataSet[:, i + 1].ravel()
            uniqueVals = set(featList)
            newGiniIndex = 0.0
            for value in uniqueVals:
                subDataSet = self.splitDataSet(dataSet, i + 1, value)
                prob = len(subDataSet) / float(len(dataSet))
                newGiniIndex += prob * self.GiniD(subDataSet)
            if (minGiniIndex > newGiniIndex):
                minGiniIndex = newGiniIndex
                bestFeature = i + 1
        return bestFeature

    def majorityCnt(self, classList):
        classCount = {}
        # classList_int = classList.astype(int)
        # max_class = np.argmax(np.bincount(classList))
        for vote in classList:
            if vote not in classCount.keys:
                classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]
        # return max_class

    def checkLabelVaule(self, dataSet, labels):
        flag = True
        tmp = dataSet[0][1:-1]
        if (tmp != dataSet[:, 1:-1]).any():
            flag = False
        return flag

    def createTree(self, dataSet, labels):
        # classList = [example[0] for example in dataSet]
        classList = dataSet[:, -1].tolist()
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        if len(dataSet[0]) == 2 or self.checkLabelVaule(dataSet, labels):
            return self.majorityCnt(classList)
        bestFeat = self.chooseBestFeatureToSplit(dataSet)
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel:{}}
        labels = np.delete(labels, bestFeat)
        featValues = dataSet[:, bestFeat].ravel()
        uniqueVals = set(featValues)
        for value in uniqueVals:
            subLabels = labels[:]
            myTree[bestFeatLabel][value] = self.createTree(self.splitDataSet(dataSet, bestFeat, value), subLabels)
        return myTree

    def createTreeByDiniIndex(self, dataSet, labels):
        # classList = [example[0] for example in dataSet]
        classList = dataSet[:, -1].tolist()
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        if len(dataSet[0]) == 2 or self.checkLabelVaule(dataSet, labels):
            return self.majorityCnt(classList)
        bestFeat = self.chooseBestFeatureToSplitByGiniIndex(dataSet)
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel:{}}
        labels = np.delete(labels, bestFeat)
        featValues = dataSet[:, bestFeat].ravel()
        uniqueVals = set(featValues)
        for value in uniqueVals:
            subLabels = labels[:]
            myTree[bestFeatLabel][value] = self.createTreeByDiniIndex(self.splitDataSet(dataSet, bestFeat, value), subLabels)
        return myTree

    def classify(self, inputTree, featLabels, testVec):
        # print(list(inputTree.keys()))
        firstStr = list(inputTree.keys())[0]
        secondDict = inputTree[firstStr]
        # print(secondDict.keys())
        featIndex = featLabels.index(firstStr)
        # print(self.X[:, -1])
        classLabel = np.argmax(np.bincount(self.X[:, -1].astype(int)))
        for key in secondDict.keys():
            if testVec[featIndex] == key:
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = self.classify(secondDict[key], featLabels, testVec)
                else:
                    classLabel = secondDict[key]
        return float(classLabel)

class CART(object):
    def __init__(self, X, labels):
        self.X = X
        self.labels = labels

    def GiniD(self, dataSet):
        numEntries = len(dataSet)
        labelCounts = {}
        label_tmp = dataSet[:, -1].astype(int)
        labelCounts_tmp = np.bincount(label_tmp)
        for label, num in enumerate(labelCounts_tmp):
            labelCounts[label] = num
        p = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key]) / numEntries
            # print prob
            p += prob**2
        return 1.0 - p

    def splitDataSet(self, dataSet, axis, value):
        retDataSet = np.concatenate((dataSet[dataSet[:,axis] == value, :axis], dataSet[dataSet[:,axis] == value, axis+1:]), axis=1)
        anotherDataSet = np.concatenate(
            (dataSet[dataSet[:, axis] != value, :axis], dataSet[dataSet[:, axis] != value, axis + 1:]), axis=1)
        return np.array(retDataSet), np.array(anotherDataSet)


    def chooseBestFeatureToSplit(self, dataSet):
        numFeatures = len(dataSet[0]) - 2
        minGiniIndex = self.GiniD(dataSet)
        bestFeature = -1
        best_value = -1
        for i in range(numFeatures):
            featList = dataSet[:, i + 1].ravel()
            uniqueVals = set(featList)
            for value in uniqueVals:
                subDataSet, anotherDataSet = self.splitDataSet(dataSet, i + 1, value)
                probSub = len(subDataSet) / float(len(dataSet))
                probAno = len(anotherDataSet) / float(len(dataSet))
                newGiniIndex = probSub * self.GiniD(subDataSet) + probAno * self.GiniD(anotherDataSet)
                if (minGiniIndex > newGiniIndex):
                    minGiniIndex = newGiniIndex
                    bestFeature = i + 1
                    bestValue = value
        return bestFeature, bestValue

    def majorityCnt(self, classList):
        classCount = {}
        max_class = np.argmax(np.bincount(classList))
        return max_class

    def checkLabelVaule(self, dataSet, labels):
        flag = True
        tmp = dataSet[0][1:-1]
        if (tmp != dataSet[:, 1:-1]).any():
            flag = False
        return flag

    def createTree(self, dataSet, labels):
        classList = dataSet[:, -1].tolist()
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        if len(dataSet[0]) == 2 or self.checkLabelVaule(dataSet, labels):
            return self.majorityCnt(classList)
        bestFeat, bestValue = self.chooseBestFeatureToSplit(dataSet)
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel:{}}
        labels = np.delete(labels, bestFeat)
        subDataSet, anotherDataSet = self.splitDataSet(dataSet, bestFeat, bestValue)
        myTree[bestFeatLabel][bestValue] = self.createTree(subDataSet, labels)
        myTree[bestFeatLabel][-bestValue] = self.createTree(anotherDataSet, labels)
        return myTree

    def classify(self, inputTree, featLabels, testVec):
        # print(list(inputTree.keys()))
        firstStr = list(inputTree.keys())[0]
        secondDict = inputTree[firstStr]
        # print(secondDict.keys())
        featIndex = featLabels.index(firstStr)
        # print(self.X[:, -1])
        # classLabel = np.argmax(np.bincount(self.X[:, -1].astype(int)))
        key = list(secondDict.keys())[0]
        # print(key)
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = self.classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
        else:
            if type(secondDict[-key]).__name__ == 'dict':
                classLabel = self.classify(secondDict[-key], featLabels, testVec)
            else:
                classLabel = secondDict[-key]
        return float(classLabel)