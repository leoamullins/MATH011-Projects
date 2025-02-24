import math
import matplotlib.pyplot as plt
import numpy as np

def h(x, g):
    return g * x * (1 - math.tanh(x))

def iterate(f, x0, n):
    x = x0
    for i in range(n):
        x = f(x)
    return x


def periodicity(y0, f, epsilon=0.0001, max_iter=1000):
    values = []
    for m in range(max_iter):
        x = iterate(f, y0, m)
        if any(abs(x - h) < epsilon for h in values):
            return m
        values.append(x)
    return -1


def generate_bifurcation_diagram(g_min=2, g_max=15, n_values=3000, preiterations=500, n_points=200):
    gammas = np.linspace(g_min, g_max, n_values)
    array_coords = np.zeros((gammas.size, n_points))
    
    # Generates coords for each gamma

    for i, g in enumerate(gammas):
        x = 0.5
        for _ in range(preiterations):
            x = h (x, g)
        for m in range(n_points): 
            x = h(x, g)
            array_coords[i, m] = x

    gamma_values = np.repeat(gammas, n_points)
    coordinates = array_coords.flatten()

    plt.figure(figsize=(10, 6))
    plt.scatter(gamma_values, coordinates, s=0.05, c=coordinates, cmap='viridis', alpha=0.5)
    plt.colorbar(label='x value')
    plt.xlabel('g')
    plt.ylabel('x')
    plt.title('Bifurcation Diagram with Improved Clarity')
    plt.show()

generate_bifurcation_diagram()




    

        



