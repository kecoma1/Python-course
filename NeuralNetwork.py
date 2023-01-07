from Layer import Layer
import numpy as np
from Data import Data
from Classifier import Classifier

class NeuralNetwork(Classifier):
    
    def __init__(self, shape, activation_function, derivative_activation_function, learning_rate):
        self.layers = []
        self.derivative_activation_function = derivative_activation_function
        self.learning_rate = learning_rate
        for i, layerSize in enumerate(shape):
            layer = Layer(layerSize, activation_function, i == 0)
            self.layers.append(layer)
            
            if i != 0:
                self.layers[i-1].connectLayer(layer)
    
    def train(self, data, epochs=1000):
        for _ in range(epochs):
            for trainRow in data.data:
                row = list(trainRow[:-1])
                row.append(1)
                nn.shoot(row)
                nn.backpropagation(trainRow[-1])

    def classify(self, data):
        predictions = []
        for testRow in data.data:
            row = list(testRow[:-1])
            row.append(1)
            
            prediction = self.shoot(row)
            predictions.append(1 if prediction > 0.5 else 0)
        print(predictions)
        
        return predictions
    
    def shoot(self, values):
        # Setting the value in the first layer
        for neuron, value in zip(self.layers[0].neurons, values):
            neuron.value = value
        
        # Shooting on every layer
        for layer in self.layers:
            layer.shoot()
        
        # Getting the value from the last layer
        return self.layers[-1].neurons[0].value

    def backpropagation(self, expected_output):
        self.delta_output(expected_output)
        self.hidden_layers_delta()
        self.increment_weights()

    def delta_output(self, expected_output):
        output = self.layers[-1].neurons[0].value
        y_sumation = self.layers[-1].neurons[0].sumation_backpropagation
        delta_output = (expected_output - output) * self.derivative_activation_function(y_sumation)
        self.layers[-1].neurons[0].delta = delta_output
    
    def hidden_layers_delta(self):
        for layer_index in range(len(self.layers))[::-1]:
            # If we are not in the first or last layer
            if layer_index == len(self.layers)-1 or layer_index == 0:
                continue
            
            for neuron in self.layers[layer_index].neurons:
                self.neuron_delta(neuron)
            
    def neuron_delta(self, neuron):
            delta_in = 0
            for connection in neuron.connections:
                delta_next_neuron = connection.ending_neuron.delta
                delta_in += connection.value*delta_next_neuron

            sumation_neuron = neuron.sumation_backpropagation
            neuron.delta = delta_in*self.derivative_activation_function(sumation_neuron)
        
    def increment_weights(self):
        for layer in self.layers:
            for neuron in layer.neurons:
                for connection in neuron.connections:
                    delta = connection.ending_neuron.delta
                    input_value = connection.starting_neuron.value
                    connection.value += self.learning_rate*delta*input_value

    def __str__(self):
        result = ""
        for layer in self.layers:
            result += str(layer)+"----------------------------------\n"
        return result


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def derivative_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

nn = NeuralNetwork((3, 2, 2, 1), sigmoid, derivative_sigmoid, 1)
d = Data("data2.csv")
nn.train(d, 1000)
predictions = nn.classify(d)

print(nn.error(predictions, d))