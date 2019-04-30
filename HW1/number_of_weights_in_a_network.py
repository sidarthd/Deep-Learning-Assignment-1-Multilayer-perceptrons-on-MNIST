import numpy as np
import matplotlib.pyplot as plt


array1 = [784, 100, 10]
array2 = [784, 50, 50, 10]
array3 = [784, 33, 33, 33, 10]

def number_of_weights(array):
    k = len(array)
    total_weights = 0
    for i in range(k-1):
        total_weights += array[i] * array[i+1]

    return total_weights


print("number of weights in the first network are", number_of_weights(array1))
print("number of weights in the first network are", number_of_weights(array2))
print("number of weights in the first network are", number_of_weights(array3))




