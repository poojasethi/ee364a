import cvxpy as cp
import numpy as np

np.random.seed(10)
(m, n) = (30, 10)
A = np.random.rand(m, n)
A = np.asmatrix(A)
b = np.random.rand(m, 1)
b = np.asmatrix(b)
c_nom = np.ones((n, 1)) + np.random.rand(n, 1)
c_nom = np.asmatrix(c_nom)


def main():
    pass


if __name__ == "__main__":
    main()
