import numpy as np
import random 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def randonacci(n):
    if n < 2:
        print("n must be atleast 2.")

    sequence = np.array([0, 1])
    prob = np.random.rand(n)
    s_n = np.where(prob <= 0.5, 1, -1)
    k = 2
    while k < n:
        k_term = sequence[k-1] + s_n[k-2] * sequence[k-2]
        sequence = np.hstack([sequence, k_term])
        k += 1
    return sequence


def new_sequence(k,m): # i.e k <= n <= m
    sequence = randonacci(m + 1)[k:]
    absolute_value = np.abs(sequence)
    indices = np.arange(k,m+1)
    grid_index = 1 / indices

    if k == 0:
        grid_index[0] = 1000000 # large number to make the value zero and avoid dividing by zero.
   
    
    return absolute_value ** grid_index
    
k = 50
m = 1000
    
xs = np.linspace(k, m, m-k+1)
ys1 = new_sequence(k, m)

ys2 = new_sequence(k, m)

ys3 = new_sequence(k, m)

limit_of_randonacci = np.array([(ys1[-1] + ys2[-1]+ ys3[-1]) / 3 for i in range(m-k+1)])

plt.plot(xs,ys1, label = "Sequence 1")
plt.plot(xs,ys2, label = "Sequence 2")
plt.plot(xs,ys3, label = "Sequence 3")
plt.plot(xs,limit_of_randonacci, label = "Limit Estimate")
plt.legend()
plt.show()

