import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

def main():
    x = cp.Variable(nonneg=True)
    y = cp.Variable(nonneg=True)
    constraints = [x >= y, y >= 1]

    cost = x + y
    prob = cp.Problem(cp.Minimize(cost), constraints)
    print(f"The problem is dcp_compliant? {prob.is_dcp()}")
    prob.solve()

    print(f"The problem status is {prob.status}")
    print(f"The optimal value p_star is {round(prob.value, 6)}")
    print(f"The optimal point x_star is {round(x.value, 6)}")
    print("*" * 100)


if __name__ == "__main__":
    main()
