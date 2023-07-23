import cvxpy as cp
import numpy as np

# data for censored fitting problem.
np.random.seed(15)

n = 20  # dimension of x's
M = 25  # number of non-censored data points
K = 100  # total number of points
c_true = np.random.randn(n, 1)
X = np.random.randn(n, K)
y = np.dot(np.transpose(X), c_true) + 0.1 * (np.sqrt(n)) * np.random.randn(K, 1)

# Reorder measurements, then censor
sort_ind = np.argsort(y.T)
y = np.sort(y.T)
y = y.T
X = X[:, sort_ind.T]
D = (y[M - 1] + y[M]) / 2.0
y = y[list(range(M))]

y = y.squeeze()
X = X.squeeze()
c_true = c_true.squeeze()


def main():
    z = cp.Variable(K - M)
    c = cp.Variable(n)

    print(f"Least squares estimate using known and filtered data")
    cost_known_points = cp.sum_squares(y[:M] - X[:, :M].T @ c)
    cost_filtered_points = cp.sum_squares(z - X[:, M:].T @ c)
    cost = cost_known_points + cost_filtered_points
    problem = cp.Problem(cp.Minimize(cost), constraints=[z >= D])
    problem.solve()
    c_hat = c.value

    print(f"Problem status: {problem.status}")
    print(f"c_hat: {c_hat}")
    print("*" * 50)

    print(f"Least squares estimate using known data only")
    cost = cost_known_points
    problem = cp.Problem(cp.Minimize(cost), constraints=[])
    problem.solve()
    c_hat_ls = c.value
    print(f"c_hat_ls: {c_hat_ls}")
    print("*" * 50)

    # Estimate from ignoring censored data.
    print(f"Relative error of c_hat: {np.linalg.norm(c_true - c_hat) / np.linalg.norm(c_true)}")
    print(f"Relative error of c_hat_ls: {np.linalg.norm(c_true - c_hat_ls) / np.linalg.norm(c_true)}")


if __name__ == "__main__":
    main()
