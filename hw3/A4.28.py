# Import packages.
import cvxpy as cp
import numpy as np

np.random.seed(0)


def main():
    n = 16  # There are 16 discrete events (possible combinations of values of X_1, X_2, X_3, X_4)

    # Each index of p represents the probability of a particular outcome.
    # 0000 represents X_1 = 0, X_2 = 0, X_3 = 0, X_4 = 0
    # 0001 represents X_1 = 0, X_2 = 0, X_3 = 0, X_4 = 1, and so on...
    # i = 0     0000
    # i = 1     0001
    # i = 2     0010
    # i = 3     0011
    # i = 4     0100
    # i = 5     0101
    # i = 6     0110
    # i = 7     0111
    # i = 8     1000
    # i = 9     1001
    # i = 10    1010
    # i = 11    1011
    # i = 12    1100
    # i = 13    1101
    # i = 14    1110
    # i = 15    1111
    p = cp.Variable(shape=n, nonneg=True)

    constraints = [
        p >= 0,
        cp.sum(p) == 1,
        (p[8] + p[9] + p[10] + p[11] + p[12] + p[13] + p[14] + p[15]) == 0.9,  # prob(X_1 = 1) = 0.9
        (p[4] + p[5] + p[6] + p[7] + p[12] + p[13] + p[14] + p[15]) == 0.9,  # prob(X_2 = 1) = 0.9
        (p[2] + p[3] + p[6] + p[7] + p[10] + p[11] + p[14] + p[15]) == 0.1,  # prob(X_3 = 1) = 0.1
        p[10] + p[14] == 0.7 * 0.1,  # prob(X_1 = 1, X_4 = 0 | X_3 = 1) = 0.7
        p[5] + p[13] == 0.6 * (p[4] + p[5] + p[12] + p[13]),  # prob(X_4 = 1 | X_2 = 1, X_3 = 0) = 0.6
    ]

    cost = p[1] + p[3] + p[5] + p[7] + p[9] + p[11] + p[13] + p[15]  # prob(X_4 = 1)

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    print("Minimizing...")
    print(f"Problem Status: {prob.status}")
    print(f"The optimal value is {prob.value}")
    print(f"The optimal probability distribution is {[round(e, 3) for e in p.value]}")

    print("*" * 50)

    prob = cp.Problem(cp.Maximize(cost), constraints)
    prob.solve()
    print("Maximizing...")
    print(f"Problem Status: {prob.status}")
    print(f"The optimal value is {prob.value}")
    print(f"The optimal probability distribution is {[round(e, 3) for e in p.value]}")
    print("*" * 50)


if __name__ == "__main__":
    main()
