from Connection import Connection

class Neuron:
    neurons = 0
    
    def __init__(self, activation_function, in_input_layer):
        self.activation_function = activation_function
        self.value = 0
        self.sumation = 0
        self.connections = []
        self.in_input_layer = in_input_layer
        
        Neuron.neurons += 1
        self.id = Neuron.neurons
    
    def connectNeuron(self, neuron):
        connection = Connection(self, neuron, 1)
        self.connections.append(connection)
    
    def connectLayer(self, layer):
        for layerNeuron in layer.neurons:
            connection = Connection(self, layerNeuron, 1)
            self.connections.append(connection)
    
    def shoot(self):
        # Setting our value
        if not self.in_input_layer:
            print(self.sumation)
            self.value = self.activation_function(self.sumation)
            self.sumation = 0
        
        # Shooting in every connection
        for connection in self.connections:
            connection.shoot()
    
    def __str__(self) -> str:
        return "Neuron "+str(self.id)