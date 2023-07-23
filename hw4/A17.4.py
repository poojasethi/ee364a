# Import packages.
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


def main():
    n = 4
    x = [0.1, 0.2, -0.05, 0.1]
    S = cp.Variable(shape=(n, n), PSD=True)

    constraints = [
        S >> 0,
        S[0][0] == 0.2,
        S[1][1] == 0.1,
        S[2][2] == 0.3,
        S[3][3] == 0.1,
        S[0][1] >= 0,
        S[0][2] >= 0,
        S[1][0] >= 0,
        S[1][2] <= 0,
        S[1][3] <= 0,
        S[2][0] >= 0,
        S[2][1] <= 0,
        S[2][3] >= 0,
        S[3][1] <= 0,
        S[3][2] >= 0,
    ]

    variance = cp.quad_form(x, S)
    prob = cp.Problem(cp.Maximize(variance), constraints)
    prob.solve()

    print(f"Problem status: {prob.status}")
    print(f"The worst case variance is {prob.value}")
    print(f"The optimal covariance matrix is:\n {np.around(S.value, decimals=5)}")
    print("*" * 50)

    constraints = [
        S >> 0,
        S[0][0] == 0.2,
        S[1][1] == 0.1,
        S[2][2] == 0.3,
        S[3][3] == 0.1,
    ]

    for i in range(n):
        for j in range(n):
            if i != j:
                constraints.append(S[i][j] == 0)

    variance = cp.quad_form(x, S)
    prob = cp.Problem(cp.Maximize(variance), constraints)
    prob.solve()

    print(f"Problem status: {prob.status}")
    print(f"The worst case variance is {prob.value}")
    print(f"The optimal covariance matrix is:\n {np.around(S.value, decimals=5)}")
    print("*" * 50)


if __name__ == "__main__":
    main()
