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


Ls = 2 ** np.linspace(1, 5, 6, dtype=int)
ams = 10. ** np.linspace(-2., 1., 7)
am_strings = [r"10^{-2}",
              r"10^{-1.5}",
              r"10^{-1}",
              r"10^{-0.5}",
              r"10^{0}",
              r"10^{0.5}",
              r"10^{1}"]
K = 26
N = 2 ** K
c_means = np.empty((len(ams), 2, L//2 + 1))
c_mses, c_mse_errs = (np.empty((len(ams), 2, L//2 + 1, K)) for _ in range(2))
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
times = np.empty(len(ams))
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










blue1 = (14./255., 115./255., 184./255.)
blue2 = (134./255., 191./255., 230./255.)
colours = []
fig, ax = plt.subplots(1, 1, figsize=(6, 3.5), constrained_layout=True)
lines = []
labels = []
ax.set_xscale("log", base=2)
ax.set_xlim(Ls[0], Ls[-1])
ax.set_xlabel(r"$L$")
ax.set_yscale("log")
ax.set_ylabel(r"$\hat{\tau}_{\hat{c}(\delta)}$")
fig.suptitle(r"Gibbs, $L\times L$")
for am, am_ in enumerate(ams):
    colours.append((blue1[0] + am*(blue2[0]-blue1[0])/(len(ams)-1),
                    blue1[1] + am*(blue2[1]-blue1[1])/(len(ams)-1),
                    blue1[2] + am*(blue2[2]-blue1[2])/(len(ams)-1)))
    lines.append(Line2D([0], [0], color=colours[am], linestyle="solid", marker="o"))
    labels.append(r"$am=%s$" % am_strings[am])
    ax.plot(Ls, c_tau_means[am], "-o", color=colours[am])
fig.legend(lines, labels, loc="center left", bbox_to_anchor=(1, .5),
           fontsize="x-small")
plt.savefig("Integrated correlation time v L (Gibbs) blue.pdf", bbox_inches="tight")
plt.show()

del fig, ax


orange1 = (1., 127./255., 0.)
orange2 = (1., 200./255., 144./255.)
colours = []
fig, ax = plt.subplots(1, 1, figsize=(6, 3.5), constrained_layout=True)
lines = []
labels = []
ax.set_xscale("log", base=2)
ax.set_xlim(Ls[0], Ls[-1])
ax.set_xlabel(r"$L$")
ax.set_yscale("log")
ax.set_ylabel(r"$\hat{\tau}_{\hat{c}(\delta)}$")
fig.suptitle(r"Gibbs, $L\times L$")
for am, am_ in enumerate(ams):
    colours.append((orange1[0] + am*(orange2[0]-orange1[0])/(len(ams)-1),
                    orange1[1] + am*(orange2[1]-orange1[1])/(len(ams)-1),
                    orange1[2] + am*(orange2[2]-orange1[2])/(len(ams)-1)))
    lines.append(Line2D([0], [0], color=colours[am], linestyle="solid", marker="o"))
    labels.append(r"$am=%s$" % am_strings[am])
    ax.plot(Ls, c_tau_means[am], "-o", color=colours[am])
fig.legend(lines, labels, loc="center left", bbox_to_anchor=(1, .5),
           fontsize="x-small")
plt.savefig("Integrated correlation time v L (Gibbs) orange.pdf", bbox_inches="tight")
plt.show()

del fig, ax
