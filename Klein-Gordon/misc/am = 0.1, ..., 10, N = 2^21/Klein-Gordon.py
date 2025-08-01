import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import time


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
    cs = np.zeros((2, L+1, N))
    c_mses, c_mse_errs = (np.empty((2, L+1, K)) for _ in range(2))
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
            for x in range(L):
                for δ in range(L+1):
                    cs[d, δ, n] += Φs[d, x] * Φs[d, (x+δ) % L]
        if n % 100000 == 0:
            print(n)
    c_means = np.mean(cs, axis=-1)
    for d in range(2):
        for δ in range(L+1):
            c = cs[d, δ]
            c_mses[d, δ, 0], c_mse_errs[d, δ, 0] = mse_err(c)
            for k in range(1, K):
                bins = int(len(c) / 2)
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
    cs = np.zeros((2, L+1, N))
    c_mses, c_mse_errs = (np.empty((2, L+1, K)) for _ in range(2))
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
    c_means = np.mean(cs, axis=-1)
    for d in range(2):
        for δ in range(L+1):
            c = cs[d, δ]
            c_mses[d, δ, 0], c_mse_errs[d, δ, 0] = mse_err(c)
            for k in range(1, K):
                bins = int(len(c) / 2)
                binned = np.empty(bins)
                for b in range(bins):
                    binned[b] = .5 * (c[2*b] + c[2*b + 1])
                c = binned
                c_mses[d, δ, k], c_mse_errs[d, δ, k] = mse_err(c)
            print(d, δ)
    return c_means, c_mses, c_mse_errs


def threedcubeGibbsCorrelation(L, am, N, disc):
    κ = (6. + (am) ** 2.) ** (-0.5)
    lattice = np.zeros((L, L, L))
    for n in range(disc):
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
                    lattice[i, j, k] = κ * (κ*γ + η)
        if n % 100000 == 0:
            print(n)
    Φs = np.zeros((3, L))
    K = int(np.log2(N))
    cs = np.zeros((3, L+1, N))
    c_mses, c_mse_errs = (np.empty((3, L+1, K)) for _ in range(2))
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
                    lattice[i, j, k] = κ * (κ*γ + η)
        for i in range(L):
            Φs[0, i] = np.sum(lattice[i])
            Φs[1, i] = np.sum(lattice[:, i])
            Φs[2, i] = np.sum(lattice[:, :, i])
        for d in range(3):
            for x in range(L):
                for δ in range(L+1):
                    cs[d, δ, n] += Φs[d, x] * Φs[d, (x+δ) % L]
        if n % 100000 == 0:
            print(n)
    c_means = np.mean(cs, axis=-1)
    for d in range(3):
        for δ in range(L+1):
            c = cs[d, δ]
            c_mses[d, δ, 0], c_mse_errs[d, δ, 0] = mse_err(c)
            for k in range(1, K):
                bins = int(len(c) / 2)
                binned = np.empty(bins)
                for b in range(bins):
                    binned[b] = .5 * (c[2*b] + c[2*b + 1])
                c = binned
                c_mses[d, δ, k], c_mse_errs[d, δ, k] = mse_err(c)
            print(d, δ)
    return c_means, c_mses, c_mse_errs


def threedcubeMetropolisCorrelation(L, am, ε, N, disc):
    κ2 = 1. / (6. + am ** 2.)
    lattice = np.zeros((L, L, L))
    for n in range(disc):
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    curr = lattice[i, j, k]
                    κ2γ = κ2 * (lattice[(i+1) % L, j, k]
                                + lattice[i, (j+1) % L, k]
                                + lattice[i, j, (k+1) % L]
                                + lattice[(i-1) % L, j, k]
                                + lattice[i, (j-1) % L, k]
                                + lattice[i, j, (k-1) % L])
                    prop = curr + np.random.uniform(-ε, ε)
                    diff = (curr - κ2γ) ** 2. - (prop - κ2γ) ** 2.
                    if (diff >= 0.) or (
                            np.random.uniform(0, 1) < np.exp(diff / (2.*κ2))):
                        lattice[i, j, k] = prop
        if n % 100000 == 0:
            print(n)
    Φs = np.zeros((3, L))
    K = int(np.log2(N))
    cs = np.zeros((3, L+1, N))
    c_mses, c_mse_errs = (np.empty((3, L+1, K)) for _ in range(2))
    for n in range(disc):
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    curr = lattice[i, j, k]
                    κ2γ = κ2 * (lattice[(i+1) % L, j, k]
                                + lattice[i, (j+1) % L, k]
                                + lattice[i, j, (k+1) % L]
                                + lattice[(i-1) % L, j, k]
                                + lattice[i, (j-1) % L, k]
                                + lattice[i, j, (k-1) % L])
                    prop = curr + np.random.uniform(-ε, ε)
                    diff = (curr - κ2γ) ** 2. - (prop - κ2γ) ** 2.
                    if (diff >= 0.) or (
                            np.random.uniform(0, 1) < np.exp(diff / (2.*κ2))):
                        lattice[i, j, k] = prop
        for i in range(L):
            Φs[0, i] = np.sum(lattice[i])
            Φs[1, i] = np.sum(lattice[:, i])
            Φs[2, i] = np.sum(lattice[:, :, i])
        for d in range(3):
            for x in range(L):
                for δ in range(L+1):
                    cs[d, δ, n] += Φs[d, x] * Φs[d, (x+δ) % L]
        if n % 100000 == 0:
            print(n)
    c_means = np.mean(cs, axis=-1)
    for d in range(3):
        for δ in range(L+1):
            c = cs[d, δ]
            c_mses[d, δ, 0], c_mse_errs[d, δ, 0] = mse_err(c)
            for k in range(1, K):
                bins = int(len(c) / 2)
                binned = np.empty(bins)
                for b in range(bins):
                    binned[b] = .5 * (c[2*b] + c[2*b + 1])
                c = binned
                c_mses[d, δ, k], c_mse_errs[d, δ, k] = mse_err(c)
            print(d, δ)
    return c_means, c_mses, c_mse_errs


methods = ["Analytic", "Gibbs\nsampler"]
colours = ["#1b9e77", "#e7298a", "#d95f02", "r", "b"]


ams = 10. ** np.linspace(-1., 1., 5)
am_strings = [r"10^{-1}", r"10^{-0.5}", r"10^{0}", r"10^{0.5}", r"10^{1}"]
K = 21
N = 2 ** K

# therms = np.empty((len(ams), int(np.ceil(N/10.))))


Ls = 2 ** np.linspace(1, 5, 5, dtype=int)
Bs = 2 ** np.arange(0, K, 1, dtype=int)
times = np.empty((len(Ls), len(ams)+1))
for L, L_ in enumerate(Ls):
    c_means = np.empty((len(ams), 2, L_+1))
    c_mses, c_mse_errs = (np.empty((len(ams), 2, L_+1, K)) for _ in range(2))
    times[L, 0] = time.time()
    for am, am_ in enumerate(ams):
        c_means[am], c_mses[am], c_mse_errs[am] = twodsquareGibbsCorrelation(
            L_, am_, N, int(.1 * N))
        for d in range(2):
            for δ in range(L_+1):
                plt.errorbar(Bs, c_mses[am, d, δ], yerr=c_mse_errs[am, d, δ],
                             color=colours[2])
                plt.xscale("log", base=2)
                plt.xlim(1, 2 ** (K-1))
                plt.xlabel(r"$B$")
                plt.ylabel(
                    r"$\widehat{\mathrm{MSE}}_{\mathrm{bin}}^{(B)}(\hat{c}_{%s}(%s))$"
                    % (d+1, δ))
                plt.title(r"Gibbs, $%s\times%s$, $am=%s$, $N=2^{%s}$" %
                          (L_, L_, am_strings[am], K))
                plt.savefig(
                    "%sx%s/Binning (Gibbs, %sx%s, am = %s, N = %s, d = %s, δ = %s).pdf"
                    % (L_, L_, L_, L_, am_, N, (d+1), δ), bbox_inches="tight")
                plt.show()
        times[L, am+1] = time.time()
    np.save("%sx%s/c_means" % (L_, L_), c_means)
    np.save("%sx%s/c_mses" % (L_, L_), c_mses)
    np.save("%sx%s/c_mse_errs" % (L_, L_), c_mse_errs)
np.save("times", times)


# Calculating true mean squared errors
c_indexes = np.empty((len(ams), 2, L+1), dtype=int)
# manually enter indexes here
for d in range(2):
    for δ in range(L+1):
        c_indexes[4, d, δ] = 8
c_tmses = np.empty((len(ams), 2, L+1))
for am, am_ in enumerate(ams):
    for d in range(2):
        # c_indexes[am, d] = np.genfromtxt(
        #     "Autocorrelation (am = %s, d = %s).csv" % (am_, d+1),
        #     delimiter=",")
        for δ in range(L+1):
            c_tmses[am, d, δ] = c_mses[am, d, δ, c_indexes[am, d, δ]]
c_taus = c_tmses / c_mses[:, :, :, 0]


# # Plotting analytic and estimated normalised sitewise correlation
# δs = np.linspace(0, L, L+1)
# analytics = np.empty((len(ams), L+1))
# for am, am_ in enumerate(ams):
#     analytics[am] = np.cosh(am_ * (δs - L/2.)) / np.cosh(am_ * L/2.)
# fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharex=True,
#                        constrained_layout=True)
# lines = []
# for m in range(len(methods)):
#     lines.append(Line2D([0], [0], color=colours[m], linestyle="solid"))
# fig.legend(lines, methods, loc="center left", bbox_to_anchor=(1, .5))
# for am, am_ in enumerate(ams):
#     ax[0, 0].set_xlim(0, L)
#     fig.suptitle(r"Gibbs, $%s\times%s$, $am=%s$, $N=2^{%s}$" %
#                  (L, L, am_, K))
#     for j in range(2):
#         ax[1, j].set_xlabel(r"$\delta$")
#         ax[1, j].set_yscale("log")
#         for i in range(2):
#             ax[i, j].plot(δs, analytics[am], color=colours[0])
#             ax[i, j].errorbar(δs, c_means[am, j]/c_means[am, j, 0],
#                               yerr=(c_tmses[am, j]**.5)/c_means[am, j, 0],
#                               color=colours[1])
#             ax[i, j].set_ylabel(r"$\tilde{c}_{%s}(\delta)$" % (j+1))
#     fig.savefig("Correlation (Gibbs, %sx%s, am = %s, N = %s).pdf" %
#                 (L, L, am_, N), bbox_inches="tight")
#     for i in range(2):
#         for j in range(2):
#             ax[i, j].clear()
# plt.close()
# del fig, ax, lines


# # Plotting estimated mean squared error and integrated correlation time
# fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True,
#                        constrained_layout=True)
# lines, labels = [], []
# for d in range(2):
#     lines.append(Line2D([0], [0], color=colours[d+3], linestyle="solid",
#                         marker="o"))
#     labels.append(r"$j=%s$" % (d+1))
# fig.legend(lines, labels, loc="center left", bbox_to_anchor=(1, .5))
# for am, am_ in enumerate(ams):
#     ax[1].set_xlim(0, L)
#     ax[1].set_xlabel(r"$\delta$")
#     ax[0].set_ylabel(r"$\widehat{\mathrm{MSE}}^*(\hat{c}_j(\delta))$")
#     ax[1].set_ylabel(r"$\hat{\tau}_{\mathrm{int},\hat{c}_j(\delta)}^*$")
#     fig.suptitle(r"Gibbs, $%s\times%s$, $am=%s$, $N=2^{%s}$" %
#                  (L, L, am_, K))
#     for d in range(2):
#         ax[0].plot(δs, c_tmses[am, d], "-o", color=colours[d+3])
#         ax[1].plot(δs, c_taus[am, d], "-o", color=colours[d+3])
#     plt.savefig(
#         "Integrated correlation time and mean squared error (Gibbs, %sx%s, am = %s, N = %s).pdf"
#         % (L, L, am_, N), bbox_inches="tight")
#     for i in range(2):
#         ax[i].clear()
# plt.close()
# del fig, ax, lines, labels
