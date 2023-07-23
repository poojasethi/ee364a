import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

n = 4
m = 2

A = np.array(
    [
        [0.95, 0.16, 0.12, 0.01],
        [-0.12, 0.98, -0.11, -0.03],
        [-0.16, 0.02, 0.98, 0.03],
        [-0.0, 0.02, -0.04, 1.03],
    ]
)

B = np.array(
    [
        [0.8, 0.0],
        [0.1, 0.2],
        [0.0, 0.8],
        [-0.2, 0.1],
    ]
)

x_init = np.ones(n)

T = 100


def draw_plots(U_val, fig_num: int) -> None:
    plt.figure(fig_num)
    plt.subplot(1, 2, 1)
    plt.scatter(U_val.T[0], U_val.T[1], label="optimal input", color="g")
    plt.title(r"Components of $u$")

    plt.subplot(1, 2, 2)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$||u_t||_2$")
    plt.scatter(np.linspace(0, T - 1, T), np.linalg.norm(U_val, 2, axis=1))
    plt.title(r"$||u_t||_2$ vs $t$")

    plt.show()


def main():
    constraints = []
    X = cp.Variable(shape=(T + 1, n))
    U = cp.Variable(shape=(T, m))

    constraints = [X[0] == x_init]
    for t in range(0, T):
        x_t = X[t]  # (n,)
        u_t = U[t]  # (m,)
        constraints.append(X[t + 1] == A @ x_t + B @ u_t)
    constraints.append(X[T] == np.zeros(n))

    # Part A
    cost = 0
    for t in range(T):
        u_t = U[t]
        cost += cp.square(cp.norm(u_t, 2))

    problem = cp.Problem(cp.Minimize(cost), constraints=constraints)
    problem.solve()

    print("Sum of squares of 2-norms")
    print(f"Problem status: {problem.status}")
    print(f"Optimal value: {problem.value}")
    print("*" * 50)

    U_val = U.value
    draw_plots(U_val, 1)

    # Part B
    cost = 0
    for t in range(T):
        u_t = U[t]
        cost += cp.norm(u_t, 2)

    problem = cp.Problem(cp.Minimize(cost), constraints=constraints)
    problem.solve()

    print("Sum of 2-norms")
    print(f"Problem status: {problem.status}")
    print(f"Optimal value: {problem.value}")
    print("*" * 50)

    U_val = U.value
    draw_plots(U_val, 2)

    # Part C
    cost = cp.max(cp.norm(U, 2, axis=1))

    problem = cp.Problem(cp.Minimize(cost), constraints=constraints)
    problem.solve()

    print("Max of 2-norms")
    print(f"Problem status: {problem.status}")
    print(f"Optimal value: {problem.value}")
    print("*" * 50)

    U_val = U.value
    draw_plots(U_val, 3)

    # Part D
    cost = 0
    for t in range(T):
        u_t = U[t]
        cost += cp.norm(u_t, 1)

    problem = cp.Problem(cp.Minimize(cost), constraints=constraints)
    problem.solve()

    print("Sum of 1-norms")
    print(f"Problem status: {problem.status}")
    print(f"Optimal value: {problem.value}")
    print("*" * 50)

    U_val = U.value
    draw_plots(U_val, 3)


if __name__ == "__main__":
    main()
