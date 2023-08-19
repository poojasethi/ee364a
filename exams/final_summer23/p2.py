import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from data.wass_midpoint_data import (
    k, d, n,
    p, q,
    C,
    plot_pdfs
)

def main():
    c_w = cp.Variable(shape=n, nonneg=True)
    X_1 = cp.Variable(shape=(n, n), nonneg=True)
    X_2 = cp.Variable(shape=(n, n), nonneg=True)

    constraints = [
        cp.sum(c_w) == 1,
        X_1 @ np.ones(n) == c_w,
        X_1.T @ np.ones(n) == p,
        X_2 @ np.ones(n) == q, 
        X_2.T @ np.ones(n) == c_w, 
    ]

    cost = cp.trace(C.T @ X_1) + cp.trace(C.T @ X_2)
    prob = cp.Problem(cp.Minimize(cost), constraints)
    print(f"The problem is dcp_compliant? {prob.is_dcp()}")
    prob.solve()

    print(f"The problem status is {prob.status}")
    print(f"The optimal objective value is {round(prob.value, 2)}")
    # print(f"The optimal c_w is {c_w.value}")

    # Plot Wasserstein midpoint
    plot_pdfs(p, q, c_w.value, title="Wasserstein midpoint")

    # Plot (algebraic) midpoint
    c_alg = 0.5 * (p + q)
    plot_pdfs(p, q, c_alg, title="Algebraic midpoint")

if __name__ == "__main__":
    main()
