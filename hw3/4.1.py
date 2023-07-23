import numpy as np
import matplotlib.pyplot as plt


# plot the feasible region
d = np.linspace(-1, 3, 500)
x, y = np.meshgrid(d, d)
plt.imshow(
    ((x >= 0) & (y >= 0) & (2 * x + y >= 1) & (x + 3 * y >= 1)).astype(int),
    extent=(x.min(), x.max(), y.min(), y.max()),
    origin="lower",
    cmap="Greys",
    alpha=0.3,
)


# plot the lines defining the constraints
x = np.linspace(0, 5, 1000)

# 2x + y >= 1
y1 = 1 - 2 * x

# x + 3y >= 1
y2 = (1 - x) / 3.0

y3 = float("inf") * np.ones(x.shape)

y4 = np.zeros(x.shape)

y5 = x

# Make plot
# plt.plot(x, np.ones_like(y1))
plt.plot(x, y1, label=r"$2x_1 + x_2 \geq 1$")
plt.plot(x, y2, label=r"$x_1 + 3x_2 \geq 1$")
plt.axvline(x=0, color="purple", label=r"$x_1 \geq 0$")
plt.plot(x, y4, label=r"$x_2 \geq 0$")
plt.plot(x, y5, label=r"$x_1 = x_2$")

plt.xlim(-0.01, 2)
plt.ylim(0, 2)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")

plt.show()
