import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def twodsquareAcceptance(L, am, ε, N, disc):
    lattice = np.zeros((L, L))
    κ2 = 1. / (4. + am ** 2.)
    for n in range(disc):
        for i in range(L):
            for j in range(L):
                curr = lattice[i, j]
                κ2γ = κ2 * (lattice[(i+1) % L, j]
                            + lattice[i, (j+1) % L]
                            + lattice[(i-1) % L, j]
                            + lattice[i, (j-1) % L])
                prop = curr + np.random.uniform(-ε, ε)
                diff = (curr - κ2γ) ** 2. - (prop - κ2γ) ** 2.
                if (diff >= 0.) or (
                        np.random.uniform(0, 1) < np.exp(diff / (2.*κ2))):
                    lattice[i, j] = prop
        if n % 10000 == 0:
            print(n)
    acc_tally = 0
    for n in range(N):
        for i in range(L):
            for j in range(L):
                curr = lattice[i, j]
                κ2γ = κ2 * (lattice[(i+1) % L, j]
                            + lattice[i, (j+1) % L]
                            + lattice[(i-1) % L, j]
                            + lattice[i, (j-1) % L])
                prop = curr + np.random.uniform(-ε, ε)
                diff = (curr - κ2γ) ** 2. - (prop - κ2γ) ** 2.
                if (diff >= 0.) or (
                        np.random.uniform(0, 1) < np.exp(diff / (2.*κ2))):
                    lattice[i, j] = prop
                    acc_tally += 1
        if n % 10000 == 0:
            print(n)
    return acc_tally / (N*L**2)


εs = [.1,
      10.**-.5,
      1.,
      10.**.2,
      10.**.5,
      10.,
      10.**1.5]
acc = np.empty(len(εs))
for ε, ε_ in enumerate(εs):
    acc[ε] = twodsquareAcceptance(16, 10.**-.5, ε_, 50000, int(.1*2**21))
