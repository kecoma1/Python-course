import numpy as np
from Data import Data
from scipy.stats import norm
from Classifier import Classifier

def distanciaEuclidea(x, y, w=None):
	"""Función para calcular la distancia euclidea
	entre 2 vectores. También se calcula la distancia euclidea
	ponderada en caso de que se pasen las ponderaciones.
	Args:
		x (numpy.array): Vector.
		y (numpy.array): Vector.
		w (numpy.array): Vector con las ponderaciones.
	Returns:
		float: Distancia euclidea
	"""
	if w is None:
		return sum([(xi-yi)**2 for xi, yi in zip(x, y)])**(1/2)
	else:
		return sum([((xi-yi)*wi)**2 for xi, yi, wi in zip(x, y, w)])**(1/2)


def distanciaManhattan(x, y):
	"""Función para calcular la distancia manhattan
	entre 2 vectores.
	Args:
		x (numpy.array): Vector.
		y (numpy.array): Vector.
	Returns:
		float: Distancia Manhattan
	"""
	return [math.abs(xi-yi) for xi, yi in zip(x, y)]

class KNN(Classifier):
    
    def __init__(self, K, distanceFunction):
        self.freq = []
        self.distanceFunction = distanceFunction
        self.nClasses = 0
        self.K = K

    def train(self, data):
        self.computeClassP(data.data)
    
    def classify(self, data):
        predictions = []
        for testRow in data.data:
            predictions.append(self.classifyRow(testRow, data.data))
        
        return predictions
    
    def classifyRow(self, row, dataArray):
        distances = []
        for dataRow in dataArray:
            distances.append((self.distanceFunction(row, dataRow), int(dataRow[-1])))
        
        return self.getHigherClass(distances)
    
    def getHigherClass(self, distances):
        sortedDistances = sorted(distances, key=lambda item: item[0])
        kShortest = sortedDistances[:self.K]
        
        classFrequency = []
        for classValue in range(self.nClasses):
            classDistances = list(filter(lambda item: item[1] == classValue, kShortest))
            if len(classDistances) == 0:
                continue
            freq = len(classDistances)
            classFrequency.append((freq, classValue))
        
        return max(classFrequency, key=lambda item: item[1])[1]
            
d = Data("data3.csv")
knn = KNN(3, distanciaEuclidea)
knn.train(d)
predictions = knn.classify(d)
print(predictions)

print(knn.error(predictions, d))
