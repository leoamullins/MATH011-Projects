import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.colors import ListedColormap

na, nb = 2000, 2000
num_iterations = 100

class Schelling:
    def __init__(self, m, n, t):
        self.grid = np.zeros((m,n))
        self.t = t
        self.randomise(na,nb)

    def unhappy(self, i, j):
        """Check if an agent is unhappy based on the threshold `t`."""
        m, n = self.grid.shape
        person_type = self.grid[i, j]

        if person_type == 0:  # Empty space
            return False
        
        i_start, i_end = max(0, i-1), min(m, i+2)
        j_start, j_end = max(0, j-1), min(n, j+2)

        neighbourhood = self.grid[i_start:i_end, j_start:j_end]
        
        # Ignore empty spaces
        occupied_neighbors = (neighbourhood != 0)
        similar_neighbors = (neighbourhood == person_type) & occupied_neighbors

        total_neighbors = np.sum(occupied_neighbors) - 1  # Exclude self
        similar_count = np.sum(similar_neighbors)

        if total_neighbors == 0:
            return False  # Avoid division by zero, no one around

        return (similar_count / total_neighbors) < (self.t / 8)  # Normalize by max neighborhood size
    def randomise(self, na, nb):
        p = np.random.rand(na + nb)
        random_values = random.sample(range(self.grid.size), na + nb)
        na_positions = random_values[:na]
        nb_positions = random_values[na:]
        
        na_indices = np.unravel_index(na_positions, self.grid.shape)
        nb_indices = np.unravel_index(nb_positions, self.grid.shape)

        self.grid[na_indices] = 1
        self.grid[nb_indices] = -1
        
    def update(self):
        empty_spots = np.argwhere(self.grid == 0).tolist()
        unhappy_people = [(i,j) for i in range(self.grid.shape[0]) for j in range(self.grid.shape[1]) if self.grid[i,j] != 0 and self.unhappy(i,j)]

        random.shuffle(unhappy_people)
        random.shuffle(empty_spots)

        for (i, j), (new_i,new_j) in zip(unhappy_people, empty_spots):
            self.grid[new_i,new_j] = self.grid[i,j]
            self.grid[i,j] = 0
        return self.grid
    
       
s = Schelling(100,100, 7)

figure, axes = plt.subplots()
plots = []
colormap = ListedColormap(['red', 'white', 'blue'])

p = axes.imshow(s.grid, interpolation='none', aspect='equal', cmap=colormap)
plots.append([p])

for i in range(num_iterations):
    s.update()
    im = axes.imshow(s.grid, interpolation="None", aspect="equal", cmap = colormap)
    plots.append([im])

anim = ArtistAnimation(figure, plots, interval = 100)
plt.show()
