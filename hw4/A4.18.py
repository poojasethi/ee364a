import cvxpy as cp
import numpy as np

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
(m, n) = (300, 100)
A = np.random.rand(m, n)
# A = np.asmatrix(A)
b = A.dot(np.ones((n, 1))) / 2
# b = np.asmatrix(b)
c = -np.random.rand(n, 1)
# c = np.asmatrix(c)

# Squeeze out extra dimensions.
b = b.squeeze()
c = c.squeeze()


def main():
    # Find a solution to the relaxed LP.
    x_rlx = cp.Variable(shape=n)
    constraints = [A @ x_rlx <= b, x_rlx >= 0, x_rlx <= 1]
    cost = c.T @ x_rlx
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

    L = problem.value

    print(f"Problem status: {problem.status}")
    print(f"The lowerbound of the cost, L, is {L}")
    print(f"The solution to the relaxation, x_rlx, is\n{np.round(x_rlx.value, decimals=3)}")

    def compute_boolean_solution(x, t: float):
        assert 0 <= t <= 1
        return np.where(x >= t, 1, 0)

    max_violations = []
    objectives = []

    t = np.linspace(0, 1, 101)

    # Carry out threshold rounding for t in [0, 1]
    for t_ in t:
        x_hat = compute_boolean_solution(x_rlx.value, t_)

        # Calculate objective value
        cost = c.T @ x_hat
        objectives.append(cost)

        # Calculate maximum constraint violation
        max_violation = max(A @ x_hat - b)
        max_violations.append(max_violation)

    objectives = np.array(objectives)
    max_violations = np.array(max_violations)

    # Point is feasible if for all x_i, Ax - b <= 0
    feasible_points = max_violations <= 0
    infeasible_points = max_violations > 0

    # Plot the maximum violation versus t.
    plt.plot(t, max_violations, label="max violation")
    plt.scatter(t[feasible_points], max_violations[feasible_points], c="g", marker="o")
    plt.scatter(t[infeasible_points], max_violations[infeasible_points], c="r", marker="x")

    # Plot the objective versus t.
    plt.plot(t, objectives, label="objective")
    plt.scatter(t[feasible_points], objectives[feasible_points], c="g", marker="o")
    plt.scatter(t[infeasible_points], objectives[infeasible_points], c="r", marker="x")

    plt.xlim(0, 1)
    plt.ylim(-60, 50)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.xlabel(r"$t$")

    plt.plot()
    plt.show()

    # Calculated associated uppper bound U.
    U = min(objectives[feasible_points])
    t_min = [t for t, o in zip(t, objectives) if o == U][0]

    print(f"A value of t for which x_hat is feasible and gives minimum objective value is {t_min}")
    print(f"The upperbound of the cost, U, is {U}")

    # Give the gap U - L
    print(f"The gap U - L between the upperbound and lowerbound on p* is {U - L}")


if __name__ == "__main__":
    main()
