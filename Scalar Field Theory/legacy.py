import numpy as np
import statistics as stat


def mse_err(array):
    """Mean squared error and mean squared error error.

    For a given array, returns the mean squared error and its corresponding
    error for the sample mean estimator.
    """
    mean, var, M = stat.fmean(array), stat.variance(array), len(array)
    return (var/M,
            (((stat.fmean((array-mean)**4.))
              - ((var**2.) * (M-3.))/(M-1.)) / (M**3.)) ** .5)


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
