import numpy as np
import random 
import matplotlib.pyplot as plt

def randonacci2(n, beta):
    if n < 2:
        print("n must be atleast 2.")

    sequence = np.array([0, 1])
    prob = np.array([(a, b) for a,b in zip(np.random.rand(n),np.random.rand(n))])
    s_n = np.where(prob[:,0] <= 0.5, 1, -1)
    t_n = np.where(prob[:,1] <= 0.5, 1, -1)
    k = 2
    while k<n:
        k_term = t_n[k-2] * sequence[k-1] + s_n[k-2] * beta * sequence[k-2]
        sequence = np.hstack([sequence, k_term])
        k += 1
    return sequence

xs = np.linspace(0,500,501)
ys1 = randonacci2(501, 0.4)
ys2 = randonacci2(501, 0.6)
ys3 = randonacci2(501, 0.8)

plt.subplot(1,3,1)
plt.plot(xs, ys1)
plt.title('Beta = 0.4')

plt.subplot(1,3,2)
plt.plot(xs, ys2)
plt.title('Beta = 0.6')

plt.subplot(1,3,3)
plt.plot(xs, ys3)
plt.title('Beta = 0.8')

plt.show()

"""
    
def beta_search(initial):
    n = 2
    beta = initial
    high = 1
    low = 0
    tolerance = 1e-6
    while n < 1000:
        sequence = randonacci2(5001, beta)
        #print(sequence[500], beta, high, low)
        if abs(sequence[5000]) > 100:
            high = beta
            beta = (beta + low) / 2
        else:
            low = beta
            beta = (beta + high) / 2
        n += 1

        if high - low < tolerance:
            break
    return beta

print(beta_search(0.5))
"""
def is_divergent(sequence, threshold=1e6):
    """Check if the sequence is divergent."""
    return np.abs(sequence[-1]) > threshold

def find_critical_beta(tolerance=1e-4, num_runs=100, sequence_length=1000):
    """Find the critical beta using binary search."""
    low = 0
    high = 1
    while high - low > tolerance:
        beta = (low + high) / 2
        divergent_count = 0
        for _ in range(num_runs):
            sequence = randonacci2(sequence_length, beta)
            if is_divergent(sequence):
                divergent_count += 1
        # If more than half of the runs diverge, assume beta > beta_c
        if divergent_count > num_runs / 2:
            high = beta
        else:
            low = beta
        print(f"Beta: {beta}, Divergent Runs: {divergent_count}/{num_runs}")
    return beta

# Find the critical beta
critical_beta = find_critical_beta()
print(f"Critical beta (transition point): {critical_beta}")