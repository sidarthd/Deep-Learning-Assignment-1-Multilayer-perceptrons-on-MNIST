import numpy as np
import matplotlib.pyplot as plt
import math


def number_of_neurons(h):
    if(h==1):
        return int(79400/794)
    else:
        
        neurons =  (-794 + math.sqrt(794**2 + 4*(h-1)*79400)) / (2 * (h-1))

    return math.floor(neurons)

print("number of neurons in each layer for the first network are", number_of_neurons(1))

print("number of neurons in each layer for the second network are", number_of_neurons(2))

print("number of neurons in each layer for the first network are", number_of_neurons(3))

print("number of neurons in each layer for the fourth network are", number_of_neurons(4))
