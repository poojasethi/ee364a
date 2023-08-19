import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from data.fit_k_markov_data import (
    T, n, K,
    Pi_train, Pi_test,
    plot_prediction_error
)

def main():
    # We're given that K = 2.
    A_1 = cp.Variable(shape=(n, n), nonneg=True)
    A_2 = cp.Variable(shape=(n, n), nonneg=True)
    
    alpha_1 = cp.Variable(nonneg=True)
    alpha_2 = cp.Variable(nonneg=True)

    one = np.ones(n)

    constraints = [
        alpha_1 + alpha_2 == 1,
        A_1.T @ one == alpha_1 * one,
        A_2.T @ one == alpha_2 * one,
    ]

    l1_distances = 0
    # Make sure to be careful about indexing!
    for t in range(K - 1, T - 1):
        pi_hat = A_1 @ Pi_train.T[t] + A_2 @ Pi_train.T[t - K + 1]
        pi = Pi_train.T[t + 1]
        l1_distances += cp.norm(pi_hat - pi, 1)

    cost = 1 / (T - K) * l1_distances

    prob = cp.Problem(cp.Minimize(cost), constraints)
    print(f"The problem is dcp_compliant? {prob.is_dcp()}")
    prob.solve()

    print(f"The problem status is {prob.status}")
    print(f"The average loss (optimal value) on the train set is {round(prob.value, 2)}")
    print(f"The optimal A_1 is\n{np.round(A_1.value, 2)}")
    print(f"The optimal A_2 is\n{np.round(A_2.value, 2)}")
    print(f"The optimal alpha_1 is: {np.round(alpha_1.value, 2)}")
    print(f"The optimal alpha_2 is: {np.round(alpha_2.value, 2)}")

    # Form the matrix Pi_hat given the optimal values.
    # Also calculate the loss on the test set.
    Pi_hat = []
    l1_distances = 0.0
    for t in range(K - 1, T - 1):
        pi_hat = A_1.value @ Pi_train.T[t] + A_2.value @ Pi_train.T[t - K + 1]
        Pi_hat.append(pi_hat)
        pi_test = Pi_test.T[t + 1]
        l1_distances += np.linalg.norm(pi_hat - pi_test, ord=1)

    Pi_hat = np.vstack(Pi_hat)
    cost = 1 / (T - K) * l1_distances

    plot_prediction_error(Pi_hat.T, Pi_train)
    plot_prediction_error(Pi_hat.T, Pi_test)
    print(f"The average loss on the test set is {round(cost, 2)}")


if __name__ == "__main__":
    main()
