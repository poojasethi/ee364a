# Import packages.
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import math

np.random.seed(0)

n = 20
pbar = np.ones((n, 1)) * 0.03 + np.r_[np.random.rand(n - 1, 1), np.zeros((1, 1))] * 0.12
S = np.random.randn(n, n)
S = np.asmatrix(S)
S = S.T * S
S = S / max(np.abs(np.diag(S))) * 0.2
S[:, -1] = np.zeros((n, 1))
S[-1, :] = np.zeros((n, 1)).T
x_unif = np.ones((n, 1)) / n
x_unit = np.asmatrix(x_unif)

# Squeeze dimensions
pbar = pbar.squeeze()
x_unif = x_unif.squeeze()


def main():
    print(f"Uniform portfolio:")
    cost = cp.quad_form(x_unif, S)
    print(f"The risk is {cost.value}")
    print("*" * 50)

    print(f"No additional constraints:")
    x = cp.Variable(shape=n)
    constraints = [
        cp.sum(x) == 1,
        cp.sum(cp.multiply(pbar, x)) == cp.sum(cp.multiply(pbar, x_unif)),
    ]

    cost = cp.quad_form(x, S)  # cost is portfolio return variance
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    print(f"Problem status: {prob.status}")
    print(f"The optimal risk is {prob.value}")
    print(f"The optimal allocation is {np.round(x.value, 3)}")
    print("*" * 50)

    print(f"Long only:")
    x = cp.Variable(shape=n)
    constraints = [
        cp.sum(x) == 1,
        cp.sum(cp.multiply(pbar, x)) == cp.sum(cp.multiply(pbar, x_unif)),
        x >= 0,
    ]
    cost = cp.quad_form(x, S)  # cost is portfolio return variance
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    print(f"Problem status: {prob.status}")
    print(f"The optimal risk is {prob.value}")
    print(f"The optimal allocation is {np.round(x.value, 3)}")
    print("*" * 50)

    print(f"Limit total short positions:")
    x = cp.Variable(shape=n)
    constraints = [
        cp.sum(x) == 1,
        cp.sum(cp.multiply(pbar, x)) == cp.sum(cp.multiply(pbar, x_unif)),
        cp.sum(cp.neg(x)) <= 0.5,
    ]
    cost = cp.quad_form(x, S)  # cost is portfolio return variance
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    print(f"Problem status: {prob.status}")
    print(f"The optimal risk is {prob.value}")
    print(f"The optimal allocation is {np.round(x.value, 3)}")
    print("*" * 50)

    # Plot the risk-return tradeoff curve.
    x = cp.Variable(shape=n)
    mu = cp.Parameter(nonneg=True)

    # Long-only portfolio
    constraints = [
        cp.sum(x) == 1,
        x >= 0,
    ]
    expected_return = pbar.T @ x
    variance = cp.quad_form(x, S)
    cost = -expected_return + mu * variance
    prob = cp.Problem(cp.Minimize(cost), constraints)

    long_return, long_variance = [], []
    for mu_val in np.logspace(-1, 5):
        mu.value = mu_val
        prob.solve()
        long_return.append(expected_return.value)
        long_variance.append(variance.value)

    long_return, long_variance = np.array(long_return), np.array(long_variance)
    long_std_deviation = np.sqrt(long_variance)

    # Limit on short portfolio
    constraints = [
        cp.sum(x) == 1,
        cp.sum(cp.neg(x)) <= 0.5,
    ]
    expected_return = pbar.T @ x
    variance = cp.quad_form(x, S)
    cost = -expected_return + mu * variance
    prob = cp.Problem(cp.Minimize(cost), constraints)

    limit_return, limit_variance = [], []
    for mu_val in np.logspace(-1, 5):
        mu.value = mu_val
        prob.solve()
        limit_return.append(expected_return.value)
        limit_variance.append(variance.value)
    limit_return, limit_variance = np.array(limit_return), np.array(limit_variance)
    limit_std_deviation = np.sqrt(limit_variance)

    plt.plot(limit_std_deviation, limit_return, label="limit shorts")
    plt.plot(long_std_deviation, long_return, label="long-only")
    plt.legend()
    plt.xlabel("standard deviation of return")
    plt.ylabel("mean return")
    plt.show()


if __name__ == "__main__":
    main()
