import cvxpy as cp
import numpy as np

# EE364a Price Bounds

r = 1.05
m = 200
n = 7

# Floor and cap for collar
F = 0.9
C = 1.15
S_0 = 1


def main():
    V = np.zeros((m, n))  # value of each asset after each scenario
    p_collar = cp.Variable()
    y = cp.Variable(shape=m, nonneg=True)

    V[:, 0] = r  # risk-free asset
    V[:, 1] = np.linspace(0.5, 2, m)  # underlying stock
    V[:, 2] = np.maximum(V[:, 1] - 1.1, 0)  # option on the exchange
    V[:, 3] = np.maximum(V[:, 1] - 1.2, 0)  # option on the exchange
    V[:, 4] = np.maximum(0.8 - V[:, 1], 0)  # option on the exchange
    V[:, 5] = np.maximum(0.7 - V[:, 1], 0)  # option on the exchange

    V[:, 6] = np.minimum(np.maximum(V[:, 1] - S_0, F - S_0), C - S_0)  # collar
    p = np.array([1, 1, 0.06, 0.03, 0.02, 0.01])

    constraints = [y >= 0, V.T @ y == cp.hstack([p, p_collar])]
    problem = cp.Problem(cp.Minimize(p_collar), constraints=constraints)
    problem.solve()

    print(f"Problem status: {problem.status}")
    print(f"Collar price lower bound: {p_collar.value}")

    problem = cp.Problem(cp.Maximize(p_collar), constraints=constraints)
    problem.solve()

    print(f"Problem status: {problem.status}")
    print(f"Collar price upper bound: {p_collar.value}")


if __name__ == "__main__":
    main()
