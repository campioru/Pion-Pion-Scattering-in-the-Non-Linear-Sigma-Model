import numpy as np
import scipy
import matplotlib.pyplot as plt

def h(rho, xs):
    return rho * ((1. - xs**2.)**.5) * np.exp(rho * xs) / (np.pi * scipy.special.iv(1, rho))

xs = np.linspace(-1., 1., 100)



plt.hist(np.genfromtxt("rho = 0.01.csv", delimiter=","), bins=100, range=(-1, 1.), density=True, color="b")
plt.plot(xs, h(.01, xs), color="r")
plt.xlim(-1., 1.)
plt.xlabel(r"$v_0$")
plt.title(r"$\rho=.01$")
plt.savefig("0.01.pdf")
plt.show()
plt.hist(np.genfromtxt("rho = 0.1.csv", delimiter=","), bins=100, range=(-1, 1.), density=True, color="b")
plt.plot(xs, h(.1, xs), color="r")
plt.xlim(-1., 1.)
plt.xlabel(r"$v_0$")
plt.title(r"$\rho=.1$")
plt.savefig("0.1.pdf")
plt.show()
plt.hist(np.genfromtxt("rho = 1.csv", delimiter=","), bins=100, range=(-1, 1.), density=True, color="b")
plt.plot(xs, h(1., xs), color="r")
plt.xlim(-1., 1.)
plt.xlabel(r"$v_0$")
plt.title(r"$\rho=1$")
plt.savefig("1.pdf")
plt.show()
plt.hist(np.genfromtxt("rho = 10.csv", delimiter=","), bins=100, range=(-1, 1.), density=True, color="b")
plt.plot(xs, h(10., xs), color="r")
plt.xlim(-1., 1.)
plt.xlabel(r"$v_0$")
plt.title(r"$\rho=10$")
plt.savefig("10.pdf")
plt.show()
