# Import packages.
import cvxpy as cp
import numpy as np

np.random.seed(0)


def main():
    x1 = cp.Variable(nonneg=True)
    x2 = cp.Variable(nonneg=True)
    constraints = [2 * x1 + x2 >= 1, x1 + 3 * x2 >= 1, x1 >= 0, x2 >= 0]

    # a
    cost = x1 + x2
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    print("Solution to 4.1a")
    print(f"The optimal value is {prob.value}")
    print(f"The optimal x1 is {x1.value} and x2 is {x2.value}")
    print("*" * 100)

    # b
    cost = -x1 - x2
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    print("Solution to 4.1b")
    print(f"The optimal value is {prob.value}")
    print(f"The optimal x1 is {x1.value} and x2 is {x2.value}")
    print("*" * 100)

    # c
    cost = x1
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    print("Solution to 4.1c")
    print(f"The optimal value is {prob.value}")
    print(f"The optimal x1 is {x1.value} and x2 is {x2.value}")
    print("*" * 100)

    # d
    cost = cp.maximum(x1, x2)
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    print("Solution to 4.1d")
    print(f"The optimal value is {prob.value}")
    print(f"The optimal x1 is {x1.value} and x2 is {x2.value}")
    print("*" * 100)

    # e
    cost = cp.square(x1) + 9 * cp.square(x2)
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    print("Solution to 4.1e")
    print(f"The optimal value is {prob.value}")
    print(f"The optimal x1 is {x1.value} and x2 is {x2.value}")
    print("*" * 100)


if __name__ == "__main__":
    main()
