import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from data.late_reporting_time_series_data import (
    N,
    y_true,
    y_tilde,
    plot_helper
)

def main():
    y_hat = cp.Variable(N, nonneg=True) # estimate of y_true
    l = cp.Variable(N-1, nonneg=True)

    constraints = [
        y_tilde[0] == y_hat[0] - l[0],
        y_tilde[N-1] == y_hat[N-1] - l[N-2], 
        cp.sum(l) <= 0.1 * cp.sum(y_hat)
    ]

    for t in range(1, N-1):
        constraints.append(y_tilde[t] == y_hat[t] - l[t] + l[t - 1])
    
    for t in range(0, N-1):
        constraints.append(l[t] <= y_hat[t])

    def neg_log_likelihood(y, T):
       # Be careful about 0-indexing. 
        l = 0
        for t in range(1, T-1):
            l += (y[t + 1] - 2 * y[t] + y[t - 1]) ** 2
        return l

    cost = neg_log_likelihood(y_hat, N)
    prob = cp.Problem(cp.Minimize(cost), constraints)
    print(f"The problem is dcp_compliant? {prob.is_dcp()}")
    prob.solve()

    print(f"The problem status is {prob.status}")
    print(f"The optimal value p_star is {round(prob.value, 6)}")
    print(f"The optimal y_hat is {y_hat.value}")
    print(f"The true y_true is {y_true}")
    print(f"The optimal l is {l.value}")

    print("*" * 100)

    def rms(y, T) -> float:
        return np.linalg.norm(y) / (T ** 0.5)

    print(f"RMS error between recovered and true time series: {rms(y_hat.value - y_true, N)}")
    print(f"RMS error between perturbed and true time series: {rms(y_tilde - y_true, N)}")

    plot_helper(y_hat.value)
    plt.show()


if __name__ == "__main__":
    main()
