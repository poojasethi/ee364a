import cvxpy as cp
import numpy as np

np.random.seed(10)

(m, n) = (30, 10)
A = np.random.rand(m, n)
A = np.asmatrix(A)
b = np.random.rand(m, 1)
# b = np.asmatrix(b)
c_nom = np.ones((n, 1)) + np.random.rand(n, 1)
# c_nom = np.asmatrix(c_nom)

# Added by Pooja
b = b.squeeze()
c_nom = c_nom.squeeze()


def main():
    print("Solve for nominal x.")
    x_nominal = cp.Variable(n)
    constraints = [A @ x_nominal >= b]
    problem = cp.Problem(cp.Minimize(c_nom.T @ x_nominal), constraints)
    problem.solve()
    print(f"Problem status: {problem.status}")
    x_nominal_val = x_nominal.value
    print(f"x_nominal: {x_nominal_val}")
    print("*" * 50)

    print("Solve for robust x.")
    F = np.r_[np.eye(n), -np.eye(n), np.ones((1, n)) / n, -np.ones((1, n)) / n]
    g = np.r_[1.25 * c_nom, -0.75 * c_nom, 1.1 * sum(c_nom) / n, -0.9 * sum(c_nom) / n]

    x_robust = cp.Variable(n)
    lmbda = cp.Variable(len(g))

    constraints = [A @ x_robust >= b, F.T @ lmbda == x_robust, lmbda >= 0]
    problem = cp.Problem(cp.Minimize(lmbda.T @ g), constraints)
    problem.solve()
    print(f"Problem status: {problem.status}")
    x_robust_val = x_robust.value
    print(f"x_robust: {x_robust_val}")
    print("*" * 50)

    print("Solve for nominal costs")
    print(f"With x_nominal {c_nom.T @ x_nominal_val}")
    print(f"With x_robust {c_nom.T @ x_robust_val}")
    print("*" * 50)

    print("Solve for worst-case cost.")
    c = cp.Variable(n)
    constraints = [F @ c <= g]
    problem = cp.Problem(cp.Maximize(c.T @ x_nominal_val), constraints)
    problem.solve()
    print(f"Problem status: {problem.status}")
    print(f"With x_nominal {problem.value}")

    problem = cp.Problem(cp.Maximize(c.T @ x_robust_val), constraints)
    problem.solve()
    print(f"Problem status: {problem.status}")
    print(f"With x_robust {problem.value}")


if __name__ == "__main__":
    main()
