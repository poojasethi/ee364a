import numpy as np
import matplotlib.pyplot as plt

# plot the feasible region
fig, ax = plt.subplots()
ax.axvspan(2, 4, alpha=0.5, color="grey")

# plot the lines defining the constraints
x = np.linspace(-5, 10, 1000)

y0 = x**2 + 1

# lamb = 0
# y1 = (1 + lamb) * pow(x, 2) - 6 * lamb * x + (8 * lamb + 1)

lamb = 1
y2 = (1 + lamb) * pow(x, 2) - 6 * lamb * x + (8 * lamb + 1)

lamb = 2
y3 = (1 + lamb) * pow(x, 2) - 6 * lamb * x + (8 * lamb + 1)

lamb = 3
y4 = (1 + lamb) * pow(x, 2) - 6 * lamb * x + (8 * lamb + 1)

# Make plot of Lagrangians
plt.figure(1)
plt.plot(x, y0, label=r"$x^2 + 1$")
# plt.plot(x, y1, label=r"$L(x, \lambda), \lambda = 0$")
plt.plot(x, y2, label=r"$L(x, \lambda), \lambda = 1$")
plt.plot(x, y3, label=r"$L(x, \lambda), \lambda = 2$")
plt.plot(x, y4, label=r"$L(x, \lambda), \lambda = 3$")
plt.plot(2, 5, "ro")

plt.xlim(-5, 10)
plt.ylim(0, 50)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")

plt.show()

# Make plot of dual
plt.figure(2)
lambs = np.linspace(-0.9, 10, 1000)
g = -((9 * lambs**2) / (1 + lambs)) + (8 * lambs + 1)
plt.plot(lambs, g)

plt.xlim(-3, 5)
plt.ylim(0, 6)
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$g(\lambda)$")

plt.show()

# Make a plot of sensitivity analysis
plt.figure(3)
u = np.linspace(-1, 8, 1000)
p = 11 + u - 6 * np.sqrt(1 + u)
plt.plot(u, p)

plt.xlim(-1, 8)
plt.ylim(-2, 10)
plt.xlabel(r"$u$")
plt.ylabel(r"$p^*(u)$")

plt.show()
