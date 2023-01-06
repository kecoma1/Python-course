from Layer import Layer

class NeuralNetwork:
    
    def __init__(self, shape, activation_function):
        self.layers = []
        for i, layerSize in enumerate(shape):
            layer = Layer(layerSize, activation_function, i == 0)
            self.layers.append(layer)
            
            if i != 0:
                self.layers[i-1].connectLayer(layer)
    
    def shoot(self, values):
        # Setting the value in the first layer
        for neuron, value in zip(self.layers[0].neurons, values):
            neuron.value = value
        
        # Shooting on every layer
        for layer in self.layers:
            layer.shoot()
        
        # Getting the value from the last layer
        return self.layers[-1].neurons[0].value

def activation_func(value):
    return 1 if value >= 0.5 else 0


nn = NeuralNetwork((2, 1), lambda value: 1 if value >= 0.5 else 0)

for layer in nn.layers:
    print(layer)
    
result = nn.shoot([3, 1])
print(result)
