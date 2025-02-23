import numpy as np
import matplotlib.pyplot as plt
import random

def random_walk(m, p):
    steps = np.where(np.random.rand(m) <= p, 1, -1) # creates a list of steps in a random walk i.e [1,-1,1,-1] etc.
    return np.concatenate(([0],np.cumsum(steps))) # adds the list of steps (cumulative sum) to the starting postion, 0. with a list of a walk.

def count_revisits(num_walks, steps, p):
    walk = np.array([random_walk(steps, p) for i in range(num_walks)])
    return sum([0 in walk[i, 1:] for i in range(num_walks)]) / num_walks


probabilities = [0.4, 0.5, 0.6]
revisit_prob = [count_revisits(1000, 1000, p) for p in probabilities]

for p, rate in zip(probabilities, revisit_prob):
    print(f"Probability of revisiting zero for p = {p} is {rate}.")

# Print OUTPUT
xs = np.linspace(0,1000,1001)
ys1 = random_walk(1000, 0.5)
ys2 = random_walk(1000, 0.7)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (10, 8))
ax1.plot(xs, ys1)
ax1.set_title('Random Walk with p=0.5')

ax2.plot(xs, ys2)
ax2.set_title('Random Walk with p=0.7')

plt.tight_layout()
plt.show()





