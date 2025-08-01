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


def SquareCorrelation(L, am, ε, N, disc):
    κ2 = 1. / (4. + (am) ** 2.)
    lattice = np.zeros((L, L))
    if ε == 0:
        κ = κ2 ** .5
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
    else:
        for n in range(disc):
            for i in range(L):
                for j in range(L):
                    curr = lattice[i, j]
                    prop = curr + np.random.uniform(-ε, ε)
                    κ2γ = κ2 * (lattice[(i-1) % L, j]
                                + lattice[(i+1) % L, j]
                                + lattice[i, (j-1) % L]
                                + lattice[i, (j+1) % L])
                    diff = (curr - κ2γ) ** 2. - (prop - κ2γ) ** 2.
                    if (diff >= 0.) or (
                            np.random.uniform(0, 1)
                            < np.exp(diff / (2.*κ2))):
                        lattice[i, j] = prop
            if n % 1000 == 0:
                print(n)
    acc_tally = 0
    K = int(.5 + np.log2(N))
    cs = np.zeros((N, 2, L//2 + 1))
    if ε == 0:
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
    else:
        for n in range(N):
            Φs = np.zeros((2, L))
            for i in range(L):
                for j in range(L):
                    curr = lattice[i, j]
                    prop = curr + np.random.uniform(-ε, ε)
                    κ2γ = κ2 * (lattice[(i-1) % L, j]
                                + lattice[(i+1) % L, j]
                                + lattice[i, (j-1) % L]
                                + lattice[i, (j+1) % L])
                    diff = (curr - κ2γ) ** 2. - (prop - κ2γ) ** 2.
                    if (diff >= 0.) or (
                            np.random.uniform(0, 1)
                            < np.exp(diff / (2.*κ2))):
                        lattice[i, j] = prop
                        Φs[0, i] += prop
                        Φs[1, j] += prop
                    else:
                        Φs[0, i] += curr
                        Φs[1, j] += curr
            for d in range(2):
                for x in range(L):
                    for δ in range(L+1):
                        cs[d, δ, n] += Φs[d, x] * Φs[d, (x+δ) % L]
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
    return np.mean(cs, axis=0), c_mses, c_mse_errs, acc_tally / (N*L**2)

Ls = 16
ams = [10. ** -.5]
am_strings = [r"10^{-0.5}"]
εs = [0,
      .1,
      10.**-.5,
      1.,
      10.**.2,
      10.**.5,
      10.,
      10.**1.5]
ε_strings = ["",
             r"10^{-1}",
             r"10^{-0.5}",
             r"10^{0}",
             r"10^{0.2}",
             r"10^{0.5}",
             r"10^{1}",
             r"10^{1.5}"]
K = 21
N = 2 ** K
c_means = np.empty((len(ams), len(εs), 2, L//2 + 1))
c_mses, c_mse_errs = (np.empty((len(ams), len(εs), 2, L//2 + 1, K)) for _ in
                      range(2))
accs = np.empty((len(ams), len(εs)))
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
        c_means[am, ε], c_mses[am, ε], c_mse_errs[am, ε], accs[am, ε] = (
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
fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey="row",
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


# Plotting estimated mean squared error and integrated correlation time against
# δ
fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharex=True,
                       constrained_layout=True)
lines = []
labels = []
for ε, ε_ in enumerate(εs):
    lines.append(Line2D([0], [0], color=colours[ε], linestyle="solid",
                        marker="o"))
    if ε == 0:
        labels.append("Gibbs")
    else:
        labels.append(r"M-H, $\epsilon=%s$" % ε_strings[ε])
fig.legend(lines, labels, loc="center left", bbox_to_anchor=(1, .5),
           fontsize="x-small")
for am, am_ in enumerate(ams):
    for d in range(2):
        for i in range(2):
            ax[i, d].set_yscale("log")
        for ε, ε_ in enumerate(εs):
            ax[0, d].plot(δs, c_tmses[am, ε, d], "-o", color=colours[ε])
            ax[1, d].plot(δs, c_taus[am, ε, d], "-o", color=colours[ε])
        ax[0, d].set_ylabel(
            r"$\widehat{\mathrm{MSE}}^*(\hat{c}_{%s}(\delta))$" % (d+1))
        ax[1, d].set_ylabel(
            r"$\hat{\tau}_{\mathrm{int},\hat{c}_{%s}(\delta)}^*$" % (d+1))
        ax[1, d].set_xlim(0, L//2)
        ax[1, d].set_xlabel(r"$\delta$")
    fig.suptitle(r"$%s\times%s$, $am=%s$, $N=2^{%s}$" %
                 (L, L, am_strings[am], K))
    plt.savefig(
        "Mean squared error and integrated correlation time v δ (%sx%s, am = %s, N = %s).pdf"
        % (L, L, am_, N), bbox_inches="tight")
    for i in range(2):
        for j in range(2):
            ax[i, j].clear()
plt.close()
del fig, ax, lines, labels


# Plotting integrated correlation time and acceptance rate against ε
c_tau_means = np.mean(c_taus, axis=(2, 3))
fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True,
                       constrained_layout=True)
lines = [Line2D([0], [0], color="r", linestyle="dashed"),
         Line2D([0], [0], color="b", linestyle="solid", marker="o")]
labels = ["Gibbs", "Metropolis-\nHastings"]
fig.legend(lines, labels, loc="center left", bbox_to_anchor=(1, .5),
           fontsize="x-small")
for am, am_ in enumerate(ams):
    ax[0].set_yscale("log")
    ax[0].set_ylabel(r"$\hat{\tau}_{\mathrm{int},\hat{c}_{j}(\delta)}^*$")
    ax[1].set_yticks([0., .25, .5, .75, 1.],
                     ["0%", "25%", "50%", "75%", "100%"])
    ax[1].set_ylabel("Acceptance rate")
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"$\epsilon$")
    if εs[0] == 0:
        ax[1].set_xlim(εs[1], εs[-1])
        ax[0].axhline(c_tau_means[am, 0], color="r", linestyle="dashed")
        ax[0].plot(εs[1:], c_tau_means[am, 1:], "-o", color="b")
        ax[1].plot(εs[1:], accs[am, 1:], "-o", color="b")
        ax[1].hlines(1., εs[1], εs[-1], color="r", linestyle="dashed")
    else:
        ax[1].set_xlim(εs[0], εs[-1])
        ax[0].plot(εs, c_tau_means[am], color="b")
        ax[1].plot(εs, accs[am], "-o", color="b")
        ax[1].hlines(1., εs[0], εs[-1], color="r", linestyle="dashed")
    fig.suptitle(r"$%s\times%s$, $am=%s$, $N=2^{%s}$" %
                 (L, L, am_strings[am], K))
    plt.savefig(
        "Integrated correlation time and acceptance rate v ε (%sx%s, am = %s, N = %s).pdf"
        % (L, L, am_, N), bbox_inches="tight")
    for i in range(2):
        ax[i].clear()
plt.close()
del fig, ax, lines, labels





orange = (1., .5, 0.)
blue = (14./255., 115./255., 184./255.)

fig, ax = plt.subplots(1, 1, figsize=(6, 3.5), constrained_layout=True)
for am, am_ in enumerate(ams):
    ax.set_xscale("log")
    ax.set_xlabel(r"$\epsilon$")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\hat{\tau}_{\hat{c}(\delta)}^*$")
    ax.set_xlim(εs[1], εs[-1])
    ax.plot(εs[1:], c_tau_means[am, 1:], "-o", color=blue, label="Metropolis-\nHastings")
    ax.axhline(c_tau_means[am, 0], color=orange, linestyle="dashed", label=r"Gibbs")
    fig.suptitle(r"$%s\times%s$, $am=%s$, $N=2^{%s}$" %
                  (L, L, am_strings[am], K))
    ax.legend()
    plt.savefig(
        "Integrated correlation time v ε (%sx%s, am = %s, N = %s).pdf" %
        (L, L, am_, N), bbox_inches="tight")
    ax.clear()
plt.close()
del fig, ax
