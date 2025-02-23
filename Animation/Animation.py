import numpy as np
import scipy.special
import random
import math
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

""" EXAMPLE
figure, axes = plt.subplots()

xs = np.linspace(0, np.pi, 51)
ys = np.sin(xs)
plots = [axes.plot(xs[:n], ys[:n], 'b-') for n in range(51)]
anim = animation.ArtistAnimation(figure, plots, interval=50)

plt.show()
"""
"""
def y(x, t, c = 1):
    G = np.exp(-1 * (np.abs(x - c * t) ** 3))
    F = np.exp(-1 * (np.abs(x + c * t) ** 3))
    return G + F

figure, axes = plt.subplots()

xs = np.linspace(-10, 10, 210)
ys = np.array([y(xs, t) for t in range(11)])


plots = [[axes.plot(xs, ys[n,:], 'b-')][0] for n in range(11)]
anim = ArtistAnimation(figure, plots, interval=100)
plt.show()
"""


def coefficients(n):
    index = np.arange(n)
    coef = 1 / scipy.special.factorial(index)
    return coef

def f(n):
    powers = (np.zeros(n) + n) ** (np.arange(n))
    poly = coefficients(n) * powers
    return np.flip(poly)

def roots(n):
    return np.roots(f(n))

figure, axes = plt.subplots()

xs = [roots(n).real for n in range(1,39)]
ys = [roots(n).imag for n in range(1,39)]

plots = [[axes.scatter(xs[n], ys[n])] for n in range(38)]

anim = ArtistAnimation(figure, plots, interval=100)
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")


plt.show()