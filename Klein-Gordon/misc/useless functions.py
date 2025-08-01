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


def twodsquareGibbsCorrelation(L, am, N, disc):
    κ = (4. + (am) ** 2.) ** (-0.5)
    lattice = np.zeros((L, L))
    for n in range(disc):
        for i in range(L):
            for j in range(L):
                γ = (lattice[(i+1) % L, j]
                     + lattice[i, (j+1) % L]
                     + lattice[(i-1) % L, j]
                     + lattice[i, (j-1) % L])
                η = np.random.normal(0., 1.)
                lattice[i, j] = κ * (κ*γ + η)
        if n % 100000 == 0:
            print(n)
    Φs = np.zeros((2, L))
    K = int(np.log2(N))
    cs = np.zeros((N, 2, L//2 + 1))
    c_mses, c_mse_errs = (np.empty((2, L//2 + 1, K)) for _ in range(2))
    for n in range(N):
        for i in range(L):
            for j in range(L):
                γ = (lattice[(i+1) % L, j]
                     + lattice[i, (j+1) % L]
                     + lattice[(i-1) % L, j]
                     + lattice[i, (j-1) % L])
                η = np.random.normal(0., 1.)
                lattice[i, j] = κ * (κ*γ + η)
        for i in range(L):
            Φs[0, i] = np.sum(lattice[i])
            Φs[1, i] = np.sum(lattice[:, i])
        for d in range(2):
            for δ in range(L//2 + 1):
                for x in range(L):
                    cs[n, d, δ] += Φs[d, x] * Φs[d, (x+δ) % L]
        if n % 100000 == 0:
            print(n)
    c_means = np.mean(cs, axis=0)
    for d in range(2):
        for δ in range(L//2 + 1):
            c = cs[:, d, δ]
            c_mses[d, δ, 0], c_mse_errs[d, δ, 0] = mse_err(c)
            for k in range(1, K):
                bins = len(c) // 2
                binned = np.empty(bins)
                for b in range(bins):
                    binned[b] = .5 * (c[2*b] + c[2*b + 1])
                c = binned
                c_mses[d, δ, k], c_mse_errs[d, δ, k] = mse_err(c)
            print(d, δ)
    return c_means, c_mses, c_mse_errs
    # return c_means, c_mses, c_mse_errs, cs[0, 0, ::10]


def twodsquareMetropolisCorrelation(L, am, ε, N, disc):
    κ2 = 1. / (4. + am ** 2.)
    lattice = np.zeros((L, L))
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
        if n % 100000 == 0:
            print(n)
    Φs = np.zeros((2, L))
    K = int(np.log2(N))
    cs = np.zeros((N, 2, L//2 + 1))
    c_mses, c_mse_errs = (np.empty((2, L//2 + 1, K)) for _ in range(2))
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
        for i in range(L):
            Φs[0, i] = np.sum(lattice[i])
            Φs[1, i] = np.sum(lattice[:, i])
        for d in range(2):
            for x in range(L):
                for δ in range(L+1):
                    cs[d, δ, n] += Φs[d, x] * Φs[d, (x+δ) % L]
        if n % 100000 == 0:
            print(n)
    c_means = np.mean(cs, axis=0)
    for d in range(2):
        for δ in range(L//2 + 1):
            c = cs[:, d, δ]
            c_mses[d, δ, 0], c_mse_errs[d, δ, 0] = mse_err(c)
            for k in range(1, K):
                bins = len(c) // 2
                binned = np.empty(bins)
                for b in range(bins):
                    binned[b] = .5 * (c[2*b] + c[2*b + 1])
                c = binned
                c_mses[d, δ, k], c_mse_errs[d, δ, k] = mse_err(c)
            print(d, δ)
    return c_means, c_mses, c_mse_errs


def twodsquareGibbs(length, am, N):
    """Gibbs sampler.

    Performs Gibbs sampling for a Klein-Gordon field over a 2D square
    lattice for a given lattice size, mass parameter, and number of iterations,
    and returns the sum of the field over each dimension at each iteration.
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
        if n % 10000. == 0.:
            print(n)
    return Φs


def twodsquareCorrelation(Φs):
    """Sitewise correlation and the binning method.

    Estimates the sitewise correlation and performs the binning method for a
    given array of field sums, returning the estimated sitewise correlation
    and the binned mean squared errors (and its corresponding error). Requires
    the number of iterations to be a power of 2 for the binning method.
    """
    N, length = np.shape(Φs)[0], np.shape(Φs)[2]
    K = int(np.log2(N))
    c_means = np.empty((2, length+1))
    c_mses, c_mse_errs = (np.empty((2, K, length+1)) for _ in range(2))
    for d in range(2):
        cs = np.zeros((N, length+1))
        for n in range(N):
            for x in range(length):
                for δ in range(length+1):
                    cs[n, δ] += Φs[n, d, x] * Φs[n, d, (x+δ) % length]
            if n % 10000 == 0:
                print(d, n)
        c_means[d] = np.mean(cs, axis=0)
        for δ in range(length+1):
            c = cs[:, δ]
            c_mses[d, 0, δ], c_mse_errs[d, 0, δ] = mse_err(c)
            for k in range(1, K):
                bins = int(len(c) / 2)
                binned = np.empty(bins)
                for b in range(bins):
                    binned[b] = .5 * (c[2*b] + c[2*b + 1])
                c = binned
                c_mses[d, k, δ], c_mse_errs[d, k, δ] = mse_err(c)
            print(d, δ)
    return c_means, c_mses, c_mse_errs


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
        if n % 100. == 0.:
            print(n)

    return Φs


def Metropolis(shape, am, N, Δ):
    """Metropolis simulation of a scalar field over a lattice (periodic)."""
    D = len(shape)
    L_total = 1
    for k in shape:
        L_total *= k
    κ2 = 1. / (2.*D + (am)**2.)
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
            φ = lattice[tuple(indices)]
            φ_ = φ + np.random.uniform(-Δ, Δ)
            ψ, ψ_ = (φ - κ2*γ) ** 2., (φ_ - κ2*γ) ** 2.
            if ψ_ <= ψ or np.random.uniform(0, 1) < np.exp((ψ - ψ_) / (2.*κ2)):
                lattice[tuple(indices)] = φ_

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
        if n % 100. == 0.:
            print(n)

    return Φs


def Correlation(Φs):
    N, D = np.shape(Φs)
    K = int(np.log2(N))
    c_means = np.empty(D, dtype=object)
    c_mses, c_mse_errs = (np.empty((D, K), dtype=object) for _ in range(2))
    for d in range(D):
        L = len(Φs[0, d])
        cs = np.zeros((N, L+1))
        for n in range(N):
            for x in range(L):
                for δ in range(L+1):
                    cs[n, δ] += Φs[n, d][x] * Φs[n, d][(x+δ) % L]
        c_means[d] = np.mean(cs, axis=0)
        for k in range(K):
            c_mses[d, k], c_mse_errs[d, k] = (np.empty(L+1) for _ in range(2))
        for δ in range(L+1):
            c = cs[:, δ]
            c_mses[d, 0][δ], c_mse_errs[d, 0][δ] = mse_err(c)
            for k in range(1, K):
                bins = int(len(c) / 2)
                binned = np.empty(bins)
                for b in range(bins):
                    binned[b] = .5 * (c[2*b] + c[2*b + 1])
                c = binned
                c_mses[d, k][δ], c_mse_errs[d, k][δ] = mse_err(c)
    return c_means, c_mses, c_mse_errs
