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


def SquareGibbsCorrelation(L, am, N, disc):
    κ2 = 1. / (4. + (am) ** 2.)
    κ = κ2 ** .5
    lattice = np.zeros((L, L))
    for n in range(disc):
        for i in range(L):
            for j in range(L):
                lattice[i, j] = np.random.normal(κ2*(
                    lattice[(i-1) % L, j]
                    + lattice[(i+1) % L, j]
                    + lattice[i, (j-1) % L]
                    + lattice[i, (j+1) % L]), κ)
        if n % 1000 == 0:
            print(n)
    K = int(.5 + np.log2(N))
    cs = np.zeros((N, 2, L//2 + 1))
    for n in range(N):
        Φs = np.zeros((2, L))
        for i in range(L):
            for j in range(L):
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
                    cs[n, d, δ] += Φs[d, x] * Φs[d, (x+δ) % L]
        if n % 1000 == 0:
            print(n)
    c_mses, c_mse_errs = (np.empty((2, L//2 + 1, K)) for _ in range(2))
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
    return np.mean(cs, axis=0), c_mses, c_mse_errs


L = 16
ams = [10. ** -.5]
am_strings = [r"10^{-0.5}"]
εs = [0]
ε_strings = [""]
K = 26
N = 2 ** K
c_means = np.empty((len(ams), len(εs), 2, L//2 + 1))
c_mses, c_mse_errs = (np.empty((len(ams), len(εs), 2, L//2 + 1, K)) for _ in
                      range(2))
Bs = 2 ** np.arange(0, K, 1, dtype=int)
colours = ["r",
           "darkorange",
           "y",
           "g",
           "b",
           "purple",
           "saddlebrown",
           "c",
           "violet",
           "dimgray"]
times = np.empty((len(ams), len(εs)))
start = time.time()
for am, am_ in enumerate(ams):
    for ε, ε_ in enumerate(εs):
        c_means[am, ε], c_mses[am, ε], c_mse_errs[am, ε] = (
            SquareCorrelation(L, am_, ε_, N, int(.1 * N)))
        for d in range(2):
            for δ in range(L//2 + 1):
                plt.errorbar(Bs, c_mses[am, ε, d, δ],
                             yerr=c_mse_errs[am, ε, d, δ], color=colours[2])
                plt.xscale("log", base=2)
                plt.xlim(1, 2 ** (K-1))
                plt.xlabel(r"$B$")
                plt.ylabel(
                    r"$\widehat{\mathrm{MSE}}_{\mathrm{bin}}^{(B)}(\hat{c}_{%s}(%s))$"
                    % (d+1, δ))
                if ε == 0:
                    plt.title(r"Gibbs, $%s\times%s$, $am=%s$, $N=2^{%s}$" %
                              (L, L, am_strings[am], K))
                    plt.savefig(
                        "Binning (Gibbs, %sx%s, am = %s, N = %s, d = %s, δ = %s).pdf"
                        % (L, L, am_, N, (d+1), δ), bbox_inches="tight")
                else:
                    plt.title(
                        r"Metropolis, $\epsilon=%s$, $%s\times%s$, $am=%s$, $N=2^{%s}$"
                        % (ε_strings[ε], L, L, am_strings[am], K))
                    plt.savefig(
                        "Binning (Metropolis, ε = %s, %sx%s, am = %s, N = %s, d = %s, δ = %s).pdf"
                        % (ε_, L, L, am_, N, (d+1), δ), bbox_inches="tight")
                plt.show()
        times[am, ε] = time.time() - start


# Calculating true mean squared errors
c_indexes = np.empty(np.shape(c_means), dtype=int)
# Manually enter indexes
c_tmses = np.empty(np.shape(c_means))
for am in range(len(ams)):
    for ε in range(len(εs)):
        for d in range(2):
            for δ in range(L//2 + 1):
                c_tmses[am, ε, d, δ] = c_mses[am, ε, d, δ,
                                              c_indexes[am, ε, d, δ]]
c_taus = c_tmses / c_mses[:, :, :, :, 0]


# Plotting analytic and estimated normalised sitewise correlation
δs = np.linspace(0, L//2, L//2 + 1)
analytics = np.empty((len(ams), L//2 + 1))
for am, am_ in enumerate(ams):
    analytics[am] = np.cosh(am_ * (δs - L/2.)) / np.cosh(am_ * L/2.)
fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharex=True,
                       constrained_layout=True)
lines = [Line2D([0], [0], color="k", linestyle="dashed")]
labels = ["Analytic"]
for ε, ε_ in enumerate(εs):
    lines.append(Line2D([0], [0], color=colours[ε], linestyle="solid"))
    if ε == 0:
        labels.append("Gibbs")
    else:
        labels.append(r"M-H, $\epsilon=%s$" % ε_strings[ε])
fig.legend(lines, labels, loc="center left", bbox_to_anchor=(1, .5),
           fontsize="x-small")
for am, am_ in enumerate(ams):
    ax[0, 0].set_xlim(0, L//2)
    fig.suptitle(r"$%s\times%s$, $am=%s$, $N=2^{%s}$" %
                 (L, L, am_strings[am], K))
    for j in range(2):
        ax[1, j].set_xlabel(r"$\delta$")
        ax[1, j].set_yscale("log")
        for i in range(2):
            ax[i, j].plot(δs, analytics[am], color="k", linestyle="dashed")
            ax[i, j].set_ylabel(r"$\tilde{c}_{%s}(\delta)$" % (j+1))
            for ε in range(len(εs)):
                ax[i, j].errorbar(δs, c_means[am, ε, j]/c_means[am, ε, j, 0],
                                  yerr=((c_tmses[am, ε, j]**.5)
                                        / c_means[am, ε, j, 0]),
                                  color=colours[ε])
    fig.savefig("Correlation (%sx%s, am = %s, N = %s).pdf" %
                (L, L, am_, N), bbox_inches="tight")
    for i in range(2):
        for j in range(2):
            ax[i, j].clear()
plt.close()
del fig, ax, lines, labels


# Plotting Gibbs integrated correlation time against am
c_tau_means = np.mean(c_taus, axis=(2, 3))
orange = (1., .5, 0.)
if εs[0] == 0:
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5), constrained_layout=True)
    ax.plot(ams, c_tau_means[:, 0] - 1., "-o", color=orange)
    ax.set_xscale("log")
    ax.set_xlim(ams[0], ams[-1])
    ax.set_xlabel(r"$am$")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\hat{\tau}_{\hat{c}(\delta)}^* - 1$")
    fig.suptitle(r"Gibbs, $%sx%s$" % (L, L))
    plt.savefig("Integrated correlation time v am (Gibbs, %sx%s).pdf" %
                (L, L), bbox_inches="tight")
    del fig, ax
