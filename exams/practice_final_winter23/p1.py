import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

import data.neural_signal_data as data

def main():
    # data.visualize_data()
    # data.plt.show()

    y = data.y
    s = data.s
    N = data.N
    T = data.N
    a_true = data.a_true

    # Part B
    a = cp.Variable(shape=N, nonneg=True)
    constraints = []

    lamb = 2
    cost = 1/T * cp.square(cp.norm(y - cp.conv(s, a).flatten(), 2)) + lamb * cp.norm(a, 1)
    prob = cp.Problem(cp.Minimize(cost), constraints)
    print(f"The problem is dcp_compliant? {prob.is_dcp()}")
    prob.solve()

    print(f"The problem status is {prob.status}")
    print(f"The optimal value p_star is {round(prob.value, 6)}")
    print(f"The optimal point a_star is {a.value}")
    print("*" * 100)

    a_hat = a.value
    # data.visualize_estimate(a_hat)
    # plt.show()

    print(f"a_true indices with nonzero entries: {data.find_nonzero_entries(a_true)}")
    a_hat_nonzero = data.find_nonzero_entries(a_hat)
    print(f"a_hat indices with nonzero entries: {a_hat_nonzero}")

    print(f"a_hat # of nonzero entries: {len(a_hat_nonzero)}")

    # Part C -- With Polishing

    tau = set(a_hat_nonzero)  # tau is the set of all indices where a_hat is greater than 0.01
    constraints = [a[i] == 0 for i in range(N) if i not in tau]

    lamb = 2
    cost = 1/T * cp.square(cp.norm(y - cp.conv(s, a).flatten(), 2))
    prob = cp.Problem(cp.Minimize(cost), constraints)
    print(f"The problem is dcp_compliant? {prob.is_dcp()}")
    prob.solve()

    print(f"The problem status is {prob.status}")
    print(f"The optimal value p_star is {round(prob.value, 6)}")
    print(f"The optimal point a_star is {a.value}")
    print("*" * 100)

    a_pol= a.value

    data.visualize_estimate(a_pol)
    plt.show()

    print(f"a_true indices with nonzero entries: {data.find_nonzero_entries(a_true)}")
    a_pol_nonzero = data.find_nonzero_entries(a_pol)
    print(f"a_pol indices with nonzero entries: {a_pol_nonzero}")

    print(f"a_pol # of nonzero entries: {len(a_pol_nonzero)}")
 


if __name__ == "__main__":
    main()
