import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from data.asymm_ls_data import *

def main():
    x = cp.Variable(shape=n)
    constraints = []

    def asymmetric_cost(kappa=1):
        cost = 0
        for i in range(m):
            a_i = A[i, :]  # Get the ith row of matrix A
            
            r = cp.sum(cp.multiply(a_i, x)) - b[i]

            if kappa >= 1:
                cost += cp.maximum(kappa * cp.square(cp.pos(r)), cp.square(r))
            else:  # kappa < 1
                # cost += cp.minimum(kappa * cp.square(cp.pos(r)), cp.square(r))  # not convex!
                cost += cp.maximum(kappa * cp.square(r), cp.square(cp.neg(r)))
        
        return cost

    kappa = 0.1
    cost = asymmetric_cost(kappa=kappa)

    print(f"Solving for {kappa=}")
    prob = cp.Problem(cp.Minimize(cost), constraints)
    print(f"The problem is dcp_compliant? {prob.is_dcp()}")
    prob.solve()

    print(f"The problem status is {prob.status}")
    print(f"The optimal value p_star is {round(prob.value, 6)}")
    print(f"The optimal value x_star is {x.value}")
    x_1 = x.value
    print()

    plt.plot()

    kappa = 1
    cost = asymmetric_cost(kappa=kappa)

    print(f"Solving for {kappa=}")
    prob = cp.Problem(cp.Minimize(cost), constraints)
    print(f"The problem is dcp_compliant? {prob.is_dcp()}")
    prob.solve()

    print(f"The problem status is {prob.status}")
    print(f"The optimal value p_star is {round(prob.value, 6)}")
    print(f"The optimal value x_star is {x.value}")
    x_2 = x.value
    print()

    kappa = 10
    cost = asymmetric_cost(kappa=kappa)

    print(f"Solving for {kappa=}")
    prob = cp.Problem(cp.Minimize(cost), constraints)
    print(f"The problem is dcp_compliant? {prob.is_dcp()}")
    prob.solve()

    print(f"The problem status is {prob.status}")
    print(f"The optimal value p_star is {round(prob.value, 6)}")
    print(f"The optimal value x_star is {x.value}")
    x_3 = x.value
    print()

    plt.hist(A @ x_1 - b, alpha=0.5, bins=10, label="kappa=0.1")
    plt.legend()
    plt.show()

    plt.hist(A @ x_2 - b, alpha=0.5, bins=10, label="kappa=1.0", color="orange")
    plt.legend()
    plt.show()

    plt.hist(A @ x_3 - b, alpha=0.5, bins=10, label="kappa=10", color="green")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
