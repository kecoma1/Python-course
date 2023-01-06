

class Connection:
    
    def __init__(self, starting_neuron, ending_neuron, value):
        self.starting_neuron = starting_neuron
        self.ending_neuron = ending_neuron
        self.value = value
    
    def shoot(self):
        result = self.value*self.starting_neuron.value
        self.ending_neuron.sumation += result
    
    def __str__(self):
        return "("+str(self.starting_neuron)+ ") -" + str(self.value) + "-> (" + str(self.ending_neuron)+")"