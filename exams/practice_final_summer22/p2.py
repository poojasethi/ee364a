import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from data.worst_case_bond_price_data import (
    y_nom,
    T,
    t,
    c,
    rho,
    kappa
)

def main():
    delta = cp.Variable(T, nonneg=True)
    constraints = [
       delta[0]  == 0,
       cp.sum(delta) == 0,
    ]

    sum_diffs = 0
    for t in range(0, T-1):
        sum_diffs += (delta[t + 1] - delta[t]) ** 2
    constraints.append(sum_diffs <= rho ** 2)

    for t in range(0, T):
        constraints.append(-kappa <= delta[t])
        constraints.append(delta[t] <= kappa)

    
    cost = 0
    for t in range(0, T):
        cost += c[t] * cp.exp(-t * (y_nom[t] + delta[t]))

    prob = cp.Problem(cp.Minimize(cost), constraints)
    print(f"The problem is dcp_compliant? {prob.is_dcp()}")
    prob.solve()

    print(f"The problem status is {prob.status}")
    print(f"The optimal value p_star is {round(prob.value, 6)}")
    print(f"The optimal delta is {delta.value}")
    print("*" * 100)


if __name__ == "__main__":
    main()
