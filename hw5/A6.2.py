import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

k = 201
t = np.linspace(-3, 3, num=k)
y = np.exp(t)
tolerance = 0.001


def main():
    a = cp.Variable(3)
    b = cp.Variable(2)
    gamma = cp.Parameter(nonneg=True)

    constraints = []
    for i in range(k):
        lhs = cp.abs(a[0] + a[1] * t[i] + a[2] * (t[i] ** 2) - y[i] * (1 + b[0] * t[i] + b[1] * (t[i] ** 2)))
        rhs = gamma * (1 + b[0] * t[i] + b[1] * (t[i] ** 2))
        constraints.append(lhs <= rhs)

    objective = cp.Minimize(0)  # feasibility problem
    problem = cp.Problem(objective, constraints)

    u = 10
    l = 0

    # Keep track of the optimal values found so far.
    a_star, b_star, gamma_star = None, None, None

    while u - l >= tolerance:
        gamma.value = (l + u) / 2
        # print(gamma.value)
        problem.solve()
        if problem.status == "optimal":
            u = gamma.value
            a_star = a.value
            b_star = b.value
            gamma_star = gamma.value
        else:
            l = gamma.value

    print(f"Optimal a's: {np.round(a_star, decimals=4)}")
    print(f"Optimal b's: {np.round(b_star, decimals=4)}")
    print(f"Optimal objective value: {gamma_star}")

    # Plot the data and optimal rational function.
    y_fit = (a_star[0] + a_star[1] * t + a_star[2] * np.power(t, 2)) / (1 + b_star[0] * t + b_star[1] * np.power(t, 2))

    plt.figure(1)
    plt.plot(t, y, label="data")
    plt.plot(t, y_fit, label="optimal rational function $f(t)$")
    plt.xlabel("$t$")
    plt.legend()
    plt.show()

    # Plot the fitting error: f(t_i) - y_i.
    plt.figure(2)
    plt.plot(t, y_fit - y, label="fitting error $f(t) - y$")
    plt.xlabel("$t$")
    plt.ylabel("fitting error")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
