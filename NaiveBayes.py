import numpy as np
from Data import Data
from scipy.stats import norm
from Classifier import Classifier

class NaiveBayes(Classifier):
    
    def __init__(self):
        self.freq = []
        self.classP = {}
        self.conditionalP = []

    def train(self, data):
        self.computeClassP(data.data)
        self.computeConditionalP(data.data, data.discreteAttributes)
    
    def classify(self, data):
        result = []
        nAttributes = data.data.shape[1]-1
        nClasses = len(np.unique(data.data[:,-1]))
        
        for testRow in data.data:
            classProbabilities = []
            
            for classValue in range(nClasses):
                probability = self.classP[classValue]
                for attributeIndex in range(nAttributes):
                    attributeValue = testRow[attributeIndex]
                    
                    if data.discreteAttributes[attributeIndex]:
                        probability *= self.conditionalP[classValue][attributeIndex][int(attributeValue)]
                    else:
                        mean, variance = self.conditionalP[classValue][attributeIndex]
                        probability *= norm(mean, variance).pdf(attributeValue)
                classProbabilities.append((classValue, probability))                        
            
            prediction = max(classProbabilities, key=lambda item: item[1])
            result.append(prediction[0])
        
        return result
        
    def computeClassP(self, dataArray):
        values, freqs = np.unique(dataArray[:,-1], return_counts=True)
        numRows = sum(freqs)
        for v, f in zip(values, freqs):
            self.freq.append((v, f))
            self.classP[int(v)] = f/numRows

    def computeConditionalP(self, dataArray, discreteAttributes):
        nAttributes = dataArray.shape[1]-1
        nClasses = len(np.unique(dataArray[:,-1]))
        
        for i in range(nClasses):
            self.conditionalP.append([])
            self.conditionalP[i] = [ [] for _ in range(nAttributes) ]
        
        for classValue in range(nClasses):
            rowsToCheck = np.where(dataArray[:,-1]==classValue)
            for attributeValue in range(nAttributes):
                if discreteAttributes[attributeValue]:
                    attributeProbabilites = self.computeAttributeProbabilities(
                        dataArray,
                        rowsToCheck[0],
                        attributeValue
                    )
                    self.conditionalP[classValue][attributeValue] = attributeProbabilites
                else:
                    meanVariance = self.computeMeanVariance(
                        dataArray,
                        rowsToCheck[0],
                        attributeValue
                    )
                    self.conditionalP[classValue][attributeValue] = meanVariance
    
    def computeMeanVariance(self, dataArray, rowIndexes, attributeValue):
        attributeValues = []
        for rowIndex in rowIndexes:
            row = dataArray[rowIndex]
            value = row[attributeValue]
            attributeValues.append(value)
            
        mean = np.mean(attributeValues)
        variance = np.var(attributeValues)
        
        return mean, variance
                
    def computeAttributeProbabilities(self, dataArray, rowIndexes, attributeValue):
        attributeValues = np.unique(dataArray[:,attributeValue])
        result = { int(attribute): 0 for attribute in attributeValues }
        
        for rowIndex in rowIndexes:
            row = dataArray[rowIndex]
            attribute = row[attributeValue]
            
            result[attribute] += 1
        
        total = sum(result.values())
        for key in result.keys():
            result[key] /= total
            
        return result
        
d = Data("data2.csv")
nb = NaiveBayes()

nb.train(d)
predictions = nb.classify(d)
error = nb.error(predictions, d)
print(str(error)+"%")