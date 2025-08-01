import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import time


def index(coord, shape):
    D = len(shape)
    i_ = coord[0]
    Π = shape[0]
    for j in range(1, D):
        i_ += coord[j] * Π
        Π *= shape[j]
    return i_


def SquareCorrelation(L, am, ε, N):
    κ2 = 1. / (4. + (am) ** 2.)
    c_means = np.zeros((2, L//2 + 1))
    lattice = np.zeros((L, L))
    if ε < 0:
        κ = κ2 ** .5
        for n in range(N):
            Φs = np.zeros((2, L))
            for j in range(L):
                for i in range(L):
                    new = np.random.normal(κ2*(
                        lattice[(i-1) % L, j]
                        + lattice[(i+1) % L, j]
                        + lattice[i, (j-1) % L]
                        + lattice[i, (j+1) % L]), κ)
                    lattice[i, j] = new
                    Φs[0, i] += new
                    Φs[1, j] += new
            for d in range(2):
                for δ in range(L//2 + 1):
                    for x in range(L):
                        c_means[d, δ] += Φs[d, x] * Φs[d, (x+δ) % L]
    else:
        for n in range(N):
            Φs = np.zeros((2, L))
            for j in range(L):
                for i in range(L):
                    curr = lattice[i, j]
                    prop = curr + np.random.uniform(-ε, ε)
                    κ2γ = κ2 * (lattice[(i-1) % L, j]
                                + lattice[(i+1) % L, j]
                                + lattice[i, (j-1) % L]
                                + lattice[i, (j+1) % L])
                    diff = (curr - κ2γ) ** 2. - (prop - κ2γ) ** 2.
                    if (diff >= 0.) or (
                            np.random.uniform(0, 1) < np.exp(diff / (2.*κ2))):
                        lattice[i, j] = prop
                        Φs[0, i] += prop
                        Φs[1, j] += prop
                    else:
                        Φs[0, i] += curr
                        Φs[1, j] += curr
            for d in range(2):
                for δ in range(L//2 + 1):
                    for x in range(L):
                        c_means[d, δ] += Φs[d, x] * Φs[d, (x+δ) % L]
    return c_means / N


def CubeCorrelation(L, am, ε, N):
    κ2 = 1. / (6. + (am) ** 2.)
    c_means = np.zeros((3, L//2 + 1))
    lattice = np.zeros((L, L, L))
    if ε < 0:
        κ = κ2 ** .5
        for n in range(N):
            Φs = np.zeros((3, L))
            for k in range(L):
                for j in range(L):
                    for i in range(L):
                        new = np.random.normal(κ2*(
                            lattice[(i-1) % L, j, k]
                            + lattice[(i+1) % L, j, k]
                            + lattice[i, (j-1) % L, k]
                            + lattice[i, (j+1) % L, k]
                            + lattice[i, j, (k-1) % L]
                            + lattice[i, j, (k+1) % L]), κ)
                        lattice[i, j, k] = new
                        Φs[0, i] += new
                        Φs[1, j] += new
                        Φs[2, k] += new
            for d in range(3):
                for δ in range(L//2 + 1):
                    for x in range(L):
                        c_means[d, δ] += Φs[d, x] * Φs[d, (x+δ) % L]
    else:
        for n in range(N):
            Φs = np.zeros((3, L))
            for k in range(L):
                for j in range(L):
                    for i in range(L):
                        curr = lattice[i, j, k]
                        prop = curr + np.random.uniform(-ε, ε)
                        κ2γ = κ2 * (lattice[(i-1) % L, j, k]
                                    + lattice[(i+1) % L, j, k]
                                    + lattice[i, (j-1) % L, k]
                                    + lattice[i, (j+1) % L, k]
                                    + lattice[i, j, (k-1) % L]
                                    + lattice[i, j, (k+1) % L])
                        diff = (curr - κ2γ) ** 2. - (prop - κ2γ) ** 2.
                        if (diff >= 0.) or (np.random.uniform(0, 1)
                                            < np.exp(diff / (2.*κ2))):
                            lattice[i, j, k] = prop
                            Φs[0, i] += prop
                            Φs[1, j] += prop
                            Φs[2, k] += prop
                        else:
                            Φs[0, i] += curr
                            Φs[1, j] += curr
                            Φs[2, k] += curr
            for d in range(3):
                for δ in range(L//2 + 1):
                    for x in range(L):
                        c_means[d, δ] += Φs[d, x] * Φs[d, (x+δ) % L]
    return c_means / N


def Correlation(shape, am, ε, N):
    D = len(shape)
    L_total = 1
    for L in shape:
        L_total *= L
    κ2 = 1. / (2.*D + (am)**2.)
    c_means = np.empty(D, dtype=object)
    for d in range(D):
        c_means[d] = np.zeros(shape[d]//2 + 1)
    Φs = np.empty(D, dtype=object)
    lattice = np.zeros(L_total)
            
    if ε < 0:
        κ = κ2 ** .5
        for n in range(N):
            for d in range(D):
                Φs[d] = np.zeros(shape[d])
            coord = np.zeros(D, dtype=int)
            for x in range(L_total):
                γ = 0.
                for d in range(D):
                    coord[d] = (coord[d]+1) % shape[d]
                    γ += lattice[index(coord, shape)]
                    coord[d] = (coord[d]-2) % shape[d]
                    γ += lattice[index(coord, shape)]
                    coord[d] = (coord[d]+1) % shape[d]
                new = np.random.normal(κ2*γ, κ)
                lattice[x] = new
                for d in range(D):
                    Φs[d][coord[d]] += new

                coord[0] += 1
                for d in range(D-1):
                    if coord[d] == shape[d]:
                        coord[d] = 0
                        coord[d+1] += 1
                    else:
                        break

            for d in range(D):
                for δ in range(shape[d]//2 + 1):
                    for x in range(shape[d]):
                        c_means[d][δ] += Φs[d][x] * Φs[d][(x+δ) % L]
    else:
        for n in range(N):
            for d in range(D):
                Φs[d] = np.zeros(shape[d])
            coord = np.zeros(D, dtype=int)
            for x in range(L_total):
                curr = lattice[x]
                prop = curr + np.random.uniform(-ε, ε)
                κ2γ = 0.
                for d in range(D):
                    coord[d] = (coord[d]+1) % shape[d]
                    κ2γ += lattice[index(coord, shape)]
                    coord[d] = (coord[d]-2) % shape[d]
                    κ2γ += lattice[index(coord, shape)]
                    coord[d] = (coord[d]+1) % shape[d]
                κ2γ *= κ2
                diff = (curr - κ2γ) ** 2. - (prop - κ2γ) ** 2.
                if (diff >= 0.) or (np.random.uniform(0, 1)
                                    < np.exp(diff / (2.*κ2))):
                    lattice[x] = prop
                    for d in range(D):
                        Φs[d][coord[d]] += prop
                else:
                    for d in range(D):
                        Φs[d][coord[d]] += curr

                coord[0] += 1
                for d in range(D-1):
                    if coord[d] == shape[d]:
                        coord[d] = 0
                        coord[d+1] += 1
                    else:
                        break

            for d in range(D):
                for δ in range(shape[d]//2 + 1):
                    for x in range(shape[d]):
                        c_means[d][δ] += Φs[d][x] * Φs[d][(x+δ) % L]
    for d in range(D):
        c_means[d] /= N
    return c_means

L = 10
am = .1
am_string = "0.1"
εs = [-1., 1.]
ε_strings = ["", "1"]
Ns = 2 ** np.linspace(4, 14, 11, dtype=int)
c_means_ps = np.empty((2, 2, 2, 6))
c_means_pc = np.empty((2, 2, 3, 6))
times_p = np.empty((2, 2, 2, len(Ns)))
colours = ["r", "b", "maroon", "navy"]

for ε, ε_ in enumerate(εs):
    for N, N_ in enumerate(Ns):
        times_p[0, 0, ε, N] = -time.time()
        temp = Correlation((L, L), am, ε_, N_)
        for i in range(2):
            c_means_ps[0, ε, i] = temp[i]
        del temp
        times_p[0, 0, ε, N] += time.time()
        print("s", "g", ε_, N_)
        times_p[0, 1, ε, N] = -time.time()
        c_means_ps[1, ε] = SquareCorrelation(L, am, ε_, N_)
        times_p[0, 1, ε, N] += time.time()
        print("s", "o", ε_, N_)

        times_p[1, 0, ε, N] = -time.time()
        temp = Correlation((L, L, L), am, ε_, N_)
        for i in range(3):
            c_means_pc[0, ε, i] = temp[i]
        del temp
        times_p[1, 0, ε, N] += time.time()
        print("c", "g", ε_, N_)
        times_p[1, 1, ε, N] = -time.time()
        c_means_pc[1, ε] = CubeCorrelation(L, am, ε_, N_)
        times_p[1, 1, ε, N] += time.time()
        print("c", "o", ε_, N_)

c_means_cs = np.empty(np.shape(c_means_ps))
c_means_cc = np.empty(np.shape(c_means_pc))
times_c = np.empty(np.shape(times_p))


fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey="row",
                       constrained_layout=True)
lines = []
labels = []
for i in range(4):
    lines.append(Line2D([0], [0], color=colours[i], linestyle="solid",
                        marker="o"))
labels.append("Python,\nmethod 1")
labels.append("Python,\nmethod 2")
labels.append("C++,\nmethod 1")
labels.append("C++,\nmethod 2")
fig.legend(lines, labels, loc="center left", bbox_to_anchor=(1, .5),
           fontsize="x-small")
ax[0, 0].set_title(r"Metropolis-Hastings, $%s\times%s$" % (L, L))
ax[1, 0].set_title(r"Metropolis-Hastings, $%s\times%s\times%s$" % (L, L, L))
ax[0, 1].set_title(r"Gibbs, $%s\times%s$" % (L, L))
ax[1, 1].set_title(r"Gibbs, $%s\times%s\times%s$" % (L, L, L))
for ε in range(2):
    ax[1, ε].set_xscale("log", base=2)
    ax[1, ε].set_xlim(Ns[0], Ns[-1])
    ax[1, ε].set_xlabel(r"$N$")
for d in range(2):
    ax[d, 0].set_yscale("log")
    ax[d, 0].set_ylabel("Runtime, s")
    for ε in range(2):
        for o in range(2):
            ax[d, ε].plot(Ns, times_p[d, o, 1-ε], "-o", color=colours[o])
        for o in range(2):
            ax[d, ε].plot(Ns, times_c[d, o, 1-ε], "-o", color=colours[o+2])
fig.suptitle(r"$am=%s$" % am_string)
plt.savefig("Runtime (am = %s).pdf" % am, bbox_inches="tight")
plt.close()
del fig, ax, lines, labels
