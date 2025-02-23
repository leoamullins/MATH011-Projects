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
    periodic = []

    # Generates coords for each gamma
    
    for i in range(gammas.size):
            coords = [iterate(lambda x: h(x, gammas[i]), 1/2, preiterations + m) for m in range(n_points)]
            array_coords[i, :] = coords

    gamma_values = [gammas[i] for i in range(gammas.size) for _ in range(n_points)]
    coordinates = array_coords.reshape((len(gamma_values),))
  

    plt.figure(figsize=(10, 6))
    plt.scatter(gamma_values, coordinates, s=0.05, c=coordinates, cmap='viridis', alpha=0.5)
    plt.colorbar(label='x value')
    plt.xlabel('g')
    plt.ylabel('x')
    plt.title('Bifurcation Diagram with Improved Clarity')
    plt.show()

generate_bifurcation_diagram()




    

        



