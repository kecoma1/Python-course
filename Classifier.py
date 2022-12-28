class Classifier:
    
    def error(self, predictions, data):
        errors = 0
        classValues = list(data.data[:,-1])
        for classValue, prediction in zip(classValues, predictions):
            if classValue != prediction:
                errors += 1
        
        return (errors / len(predictions)) * 100