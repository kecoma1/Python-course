from Neuron import Neuron

class Layer:
    
    def __init__(self, num_neurons, activation_function, input_layer):
        self.input_layer = input_layer
        self.neurons = []
        for _ in range(num_neurons):
            n = Neuron(activation_function, input_layer)
            self.neurons.append(n)

    def shoot(self):
        for neuron in self.neurons:
            neuron.shoot()
    
    def connectLayer(self, layer):
        for neuron in self.neurons:
            neuron.connectLayer(layer)
    
    def connectNeuron(self, neuron):
        for neuron in self.neurons:
            neuron.connectNeuron(neuron)
    
    def __str__(self):
        result = ""
        for neuron in self.neurons:
            result += str(neuron)+":\n"
            for connection in neuron.connections:
                result += "\t"+str(connection)+"\n"
        return result