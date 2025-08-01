import numpy as np
import matplotlib.pyplot as plt
import time


def twodsquareGibbs(L, am, N):
    κ = (4. + (am) ** 2.) ** (-0.5)
    lattice = np.zeros((L, L))
    Φs = np.zeros((N, 2, L))
    for n in range(N):
        for i in range(L):
            for j in range(L):
                γ = (lattice[(i+1) % L, j]
                     + lattice[i, (j+1) % L]
                     + lattice[(i-1) % L, j]
                     + lattice[i, (j-1) % L])
                η = np.random.normal(0., 1.)
                lattice[i, j] = κ * (κ*γ + η)
        for q in range(L):
            Φs[n, 0, q] = np.sum(lattice[q])
            Φs[n, 1, q] = np.sum(lattice[:, q])
        if n % 1000. == 0.:
            print(n)
    return Φs


def threedcubeGibbs(L, am, N):
    κ = (6. + (am) ** 2.) ** (-0.5)
    lattice = np.zeros((L, L, L))
    Φs = np.zeros((N, 3, L))
    for n in range(N):
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    γ = (lattice[(i+1) % L, j, k]
                         + lattice[i, (j+1) % L, k]
                         + lattice[i, j, (k+1) % L]
                         + lattice[(i-1) % L, j, k]
                         + lattice[i, (j-1) % L, k]
                         + lattice[i, j, (k-1) % L])
                    η = np.random.normal(0., 1.)
                    lattice[i, j] = κ * (κ*γ + η)
        for q in range(L):
            Φs[n, 0, q] = np.sum(lattice[q])
            Φs[n, 1, q] = np.sum(lattice[:, q])
            Φs[n, 2, q] = np.sum(lattice[:, :, q])
        if n % 1000. == 0.:
            print(n)
    return Φs


def Gibbs(shape, am, N):
    """Gibbs simulation of a scalar field over a lattice (periodic)."""
    D = len(shape)
    L_total = 1
    for k in shape:
        L_total *= k
    κ = (2.*D + (am)**2.) ** (-0.5)
    lattice = np.zeros(shape)
    Φs = np.empty((N, D), dtype=object)
    for n in range(N):
        indices = np.zeros(D, dtype=int)
        for x in range(L_total):
            γ = 0.
            for d in range(D):
                L = shape[d]
                indices_ = np.delete(indices, d)
                index = indices[d]
                lattice_ = np.moveaxis(lattice, d, -1)[tuple(indices_)]
                γ += lattice_[(index-1) % L] + lattice_[(index+1) % L]
            η = np.random.normal(0., 1.)
            lattice[tuple(indices)] = κ * (κ*γ + η)

            indices[0] += 1
            for d in range(D-1):
                if indices[d] == shape[d]:
                    indices[d] = 0
                    indices[d+1] += 1

        for d in range(D):
            L = shape[d]
            Φs[n, d] = np.empty(L)
            lattice_ = np.moveaxis(lattice, d, 0)
            for x in range(L):
                Φs[n, d][x] = np.sum(lattice_[x])
        if n % 1000. == 0.:
            print(n)

    return Φs


BeforeQuickSquare = time.time()
twodsquareGibbs(10, 1., 5000)
AfterQuickSquare = time.time()
QuickSquare = AfterQuickSquare - BeforeQuickSquare
BeforeSlowSquare = time.time()
Gibbs((10, 10), 1., 5000)
AfterSlowSquare = time.time()
SlowSquare = AfterSlowSquare - BeforeSlowSquare

BeforeQuickCube = time.time()
threedcubeGibbs(10, 1., 5000)
AfterQuickCube = time.time()
QuickCube = AfterQuickCube - BeforeQuickCube
BeforeSlowCube = time.time()
Gibbs((10, 10, 10), 1., 5000)
AfterSlowCube = time.time()
SlowCube = AfterSlowCube - BeforeSlowCube

plt.plot([0, 5000], [0, SlowSquare], color="r", label=r"$10\times10$ (general)")
plt.plot([0, 5000], [0, QuickSquare], color="r", linestyle="dashed", label=r"$10\times10$ (optimised)")
plt.xlim(0, 5000)
plt.xlabel(r"$N$")
plt.ylabel("Time to simulate (seconds)")
plt.title(r"Gibbs, $am=1$")
plt.legend()
plt.savefig("square.png", bbox_inches="tight")
plt.show()

plt.plot([0, 5000], [0, SlowCube], color="b", label=r"$10\times10\times10$ (general)")
plt.plot([0, 5000], [0, QuickCube], color="b", linestyle="dashed", label=r"$10\times10\times10$ (optimised)")
plt.xlim(0, 5000)
plt.xlabel(r"$N$")
plt.ylabel("Time to simulate (seconds)")
plt.title(r"Gibbs, $am=1$")
plt.legend()
plt.savefig("cube.png", bbox_inches="tight")
plt.show()
