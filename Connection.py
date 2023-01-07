from random import uniform

class Connection:
    
    def __init__(self, starting_neuron, ending_neuron):
        self.starting_neuron = starting_neuron
        self.ending_neuron = ending_neuron
        self.value = uniform(-1, 1)
    
    def shoot(self):
        result = self.value*self.starting_neuron.value
        self.ending_neuron.sumation += result
    
    def __str__(self):
        return "("+str(self.starting_neuron)+ ") -" + str(self.value) + "-> (" + str(self.ending_neuron)+")"