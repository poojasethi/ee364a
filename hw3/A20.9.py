# Import packages.
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from typing import Any, Tuple, Optional

np.random.seed(0)


def get_storage_tradeoff_data(draw_plot: bool = False) -> Tuple[int, Any, Any, Any]:
    T = 96
    t = np.linspace(1, T, num=T).reshape(T, 1)
    p = np.exp(-np.cos((t - 15) * 2 * np.pi / T) + 0.01 * np.random.randn(T, 1))
    u = 2 * np.exp(-0.6 * np.cos((t + 40) * np.pi / T) - 0.7 * np.cos(t * 4 * np.pi / T) + 0.01 * np.random.randn(T, 1))

    if draw_plot:
        plt.figure(1)
        plt.plot(t / 4, p)
        plt.plot(t / 4, u, "r")
        plt.show()

    return T, t.squeeze(), p.squeeze(), u.squeeze()


T, t, p, u = get_storage_tradeoff_data()


def optimize_battery(Q: int = 35, C: int = 3, D: int = 3, draw_plot: bool = False) -> Optional[float]:
    c = cp.Variable(shape=T)
    q = cp.Variable(shape=T, nonneg=True)

    constraints = []

    for i in range(T):
        constraints += [
            q[i] <= Q,
            q[i] >= 0,
            c[i] <= C,
            c[i] >= -D,
        ]

        if i == 0:
            constraints += [q[i] == q[T - 1] + c[T - 1]]
        elif i >= 1:
            constraints += [q[i] == q[i - 1] + c[i - 1]]

    cost = p @ (u + c)  # cost
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    print(f"Problem Status: {prob.status}")
    print(f"The optimal cost is {prob.value}")
    print(f"The optimal c (charging of the battery) is {c.value}")
    print(f"The optimal q (nonnegative energy stored in the battery) is {q.value}")

    # Plot  u, p, c, and q vs. t
    if draw_plot:
        plt.figure(1)
        plt.plot(t / 4, u, label="u")
        plt.plot(t / 4, p, label="p")
        plt.plot(t / 4, c.value, label="c")
        plt.plot(t / 4, q.value, label="q")

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.xlabel("t (as hour of the day)")
        plt.show()

    return prob.value


if __name__ == "__main__":
    # 20.9b
    optimize_battery(Q=35, C=3, D=3, draw_plot=True)

    # 20.9.c
    # Q = np.arange(200)

    # C = D = 3
    # minimum_costs = []
    # for _Q in Q:
    #     minimum_costs.append(optimize_battery(Q=_Q, C=C, D=D, draw_plot=False))

    # plt.plot(Q, minimum_costs, label=f"C={C}, D={D}")

    # C = D = 1
    # minimum_costs = []
    # for _Q in Q:
    #     minimum_costs.append(optimize_battery(Q=_Q, C=C, D=D, draw_plot=False))

    # plt.plot(Q, minimum_costs, label=f"C={C}, D={D}")

    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    # plt.xlabel("Q")
    # plt.ylabel("Minimum total cost ($)")

    # plt.title(f"Minimal total cost vs. storage capacity")

    # plt.show()
