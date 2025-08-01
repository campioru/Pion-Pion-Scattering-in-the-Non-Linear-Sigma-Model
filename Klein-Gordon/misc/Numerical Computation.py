"""Pion-Pion Scattering in the Non-Linear Sigma Model: Numerical Computation.

Simulates a Klein-Gordon field over a 2D square lattice using the Gibbs
sampler, estimates the sitewise correlation function, and compares the
estimated and analytic normalised sitewise correlation functions.

@author: Ruaidhrí Campion
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def twodsquareGibbs(length, am, N):
    """Gibbs sampler and sitewise correlation.

    Performs Gibbs sampling for a Klein-Gordon field over a 2D square lattice
    for a given lattice size, mass parameter, and number of iterations, and
    returns the estimated sitewise correlation over both dimensions.
    """
    κ = (4. + (am) ** 2.) ** (-0.5)
    lattice = np.zeros((length, length))
    Φs = np.zeros((N, 2, length))
    for n in range(N):
        for i in range(length):
            for j in range(length):
                γ = (lattice[(i+1) % length, j]
                     + lattice[i, (j+1) % length]
                     + lattice[(i-1) % length, j]
                     + lattice[i, (j-1) % length])
                η = np.random.normal(0., 1.)
                lattice[i, j] = κ * (κ*γ + η)
        for i in range(length):
            Φs[n, 0, i] = np.sum(lattice[i])
        for j in range(length):
            Φs[n, 1, j] = np.sum(lattice[:, j])
        if n % 10000 == 0:
            print("Phi", n)
    c_means = np.empty((2, length+1))
    for d in range(2):
        cs = np.zeros((N, length+1))
        for n in range(N):
            for x in range(length):
                for δ in range(length+1):
                    cs[n, δ] += Φs[n, d, x] * Φs[n, d, (x+δ) % length]
            if n % 10000 == 0:
                print("Correlation", d, n)
        c_means[d] = np.mean(cs, axis=0)
    return c_means


# System parameters
length = 32
am = .5
K = 22
N = 2 ** K


# Simulating and calculating sitewise correlation (N.B. this takes a very long
# time to compute for large N)
c_means = twodsquareGibbs(length, am, N)


# Plotting analytic and estimated normalised sitewise correlation
δs = np.linspace(0, length, length+1)
analytic = np.cosh(am * (δs - length/2.)) / np.cosh(am * length/2.)
fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharex=True,
                       constrained_layout=True)
ax[0, 0].set_xlim(0, length)
fig.suptitle(r"Gibbs, $%s\times%s$, $am=%s$, $N=2^{%s}$" %
             (length, length, am, K))
methods = ["Analytic", "Gibbs\nsampler"]
colours = ["#1b9e77", "#e7298a"]
lines = []
for m in range(len(methods)):
    lines.append(Line2D([0], [0], color=colours[m], linestyle="solid"))
fig.legend(lines, methods, loc="center left", bbox_to_anchor=(1, .5))
for j in range(2):
    ax[1, j].set_xlabel(r"$\delta$")
    ax[1, j].set_yscale("log")
    for i in range(2):
        ax[i, j].plot(δs, analytic, color=colours[0])
        ax[i, j].plot(δs, c_means[j]/c_means[j][0], color=colours[1])
        ax[i, j].set_ylabel(r"$\tilde{c}_{%s}(\delta)$" % (j+1))
fig.savefig("Correlation (Gibbs, %sx%s, am = %s, N = %s).pdf" %
            (length, length, am, N), bbox_inches="tight")
