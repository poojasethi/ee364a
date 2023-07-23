import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import math

K = 10
N = 10 * K  # rule-of-thumb

t = np.linspace(-math.pi + math.pi / N, math.pi, 2 * N)  # (2N,)
y = np.where(np.abs(t) <= math.pi / 2, 1, 0)  # (2N,)

A = np.zeros((2 * N, K + 1))  # (2N, K+1)
for i in range(0, 2 * N):
    for k in range(1, K + 1):
        A[i][k] = k * t[i]
A = np.cos(A)


def main():
    # Find the optimal L2 approximation
    x_2, _, _, _ = np.linalg.lstsq(A, y)  # solve for coefficients
    y_2 = A @ x_2

    # Find the optimal L1 approximation
    x_1 = cp.Variable(shape=(K + 1))
    cost = cp.norm(A @ x_1 - y, 1)
    problem = cp.Problem(cp.Minimize(cost))
    problem.solve()
    y_1 = A @ x_1.value

    print(f"Problem status: {problem.status}")
    print(f"Problem : {problem.status}")

    plt.figure(1)
    plt.plot(t, y, label="$y$")
    plt.plot(t, y_1, label="$l1$ approximation")
    plt.plot(t, y_2, label="$l2$ approximation")
    plt.legend()
    plt.show()

    plt.figure(2)
    l1_counts, l1_bins = np.histogram(y - y_1, bins=40, range=(-1, 1))
    plt.stairs(l1_counts, l1_bins, label="$l1$ residuals")

    l2_counts, l2_bins = np.histogram(y - y_2, bins=40, range=(-1, 1))
    plt.stairs(l2_counts, l2_bins, label="$l2$ residuals")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
