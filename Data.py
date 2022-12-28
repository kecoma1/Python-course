import pandas as pd
import numpy as np

class Data:
    
    def __init__(self, filename, allDiscrete=False):
        self.discreteAttributes = []
        self.dictionary = {}
        self.data = None
        
        df = pd.read_csv(filename)
        
        if allDiscrete:
            self.discreteAttributes = [ True for _ in df.dtypes ]
        else:
            self.assignDiscreteAttributes(df.dtypes)
        
        self.buildDictionary(df)
        
        self.buildData(df)
    
    def assignDiscreteAttributes(self, types):
        # Iterate with a loop over the types
        for t in types:
            self.discreteAttributes.append(True if t == object else False)

    def buildDictionary(self, df):
        for item, discrete in zip(df.iteritems(), self.discreteAttributes):
            if not discrete:
                continue
            columnName = item[0]
            possibleValues = list(df[columnName].unique()).copy()
            possibleValues.sort()
            
            # Filling the dictionary
            self.dictionary[columnName] = { value: index for index, value in enumerate(possibleValues) }
    
    def buildData(self, df):
        self.data = np.zeros(shape=df.shape)
        for i, row in enumerate(df.iterrows()):
            for n, column in enumerate(row[1].items()):
                self.data[i][n] = column[1] if not self.discreteAttributes[n] else self.dictionary[column[0]][column[1]]
