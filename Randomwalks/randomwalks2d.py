import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def random_walk_2d(n):
    p = np.random.rand(n) # Generate a list of values form 0 to 1 which we will use for probability
    
    steps = np.zeros((n,2)) # creates an array which we will add steps to
    """
    Using a boolean mask on the p-list to then change the elements (steps) of the zero matrix created above
    """
    steps[p < 0.25] = (0, 1)  
    steps[(0.25 <= p) & (p < 0.5)] = (0, -1)
    steps[(0.5 <= p) & (p < 0.75)] = (1, 0)
    steps[(0.75 <= p)] = (-1, 0)
    """
   To sum and track the position from each step we cumulatively sum up the steps, and then add the starting position at the top of the total path. 
    """
    total_path = np.vstack([[0,0],np.cumsum(steps, axis = 0)]) # has size (n+1,2)
    return total_path

"Plotting the walk in an x-y plane"
num_walks = 500
steps = 1000
path = random_walk_2d(steps)
xs = path[:,0]
ys = path[:,1]

plt.figure(figsize=(8, 8))
plt.plot(xs, ys, marker='o', markersize=1, linestyle='-', color='b')
plt.grid(True)
plt.title("2D Random Walk")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()



def distance_from_origin(num_walks, steps):
    """Calculate the disstance away from the origin, returning a matrix whose entries represent a distance from the origin at any given step in each walk"""
    # Creating an array of num_walks many walks
    walk = np.array([random_walk_2d(steps) for i in range(num_walks)]) 

    # Summing the squared values of each x,y position in a walk (axis = 2), removing the 3rd dimension.
    sum_values = np.sum(walk**2, axis = 2) 

    # Square-rooting the squared values giving a distance
    magnitude = np.sqrt(sum_values)
    
    return magnitude

def avg_dist(num_walks, steps):
    ''' Calculating the mean distance from the origin of the ith step (axis = 0) of a walk'''
    return np.sum(distance_from_origin(num_walks, steps), axis = 0) / num_walks 

def model(x, c, e):
    return c * (x ** e)

"""Plotting the average distance away for each step in a walk"""
xs2 = np.linspace(0,steps,steps + 1)
ys2 = avg_dist(num_walks, steps)
plt.plot(xs2, ys2)
plt.show()
    

'''Finding a model'''
popt, pcov = curve_fit(model, xs2, ys2, p0 = [1, 1])
print("Optimised parameters:", popt)

plt.plot(xs2, ys2, label = "Original Data")
plt.plot(xs2, model(xs2, *popt), 'r-', label = "Fitted Curve")
plt.legend()
plt.show()

    
            

