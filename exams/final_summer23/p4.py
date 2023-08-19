import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from data.rob_logistic_reg_data import *

def main():
    print("Fitting logistic regression model...")
    theta = cp.Variable(shape=d)

    l = 0.0
    for i in range(n):
        y_i = y[i] # (1,)
        x_i = X[i] # (d,)
        l += cp.logistic(-1 * y_i * theta.T @ x_i)

    prob = cp.Problem(cp.Minimize(l))
    print(f"The problem is dcp_compliant? {prob.is_dcp()}")
    prob.solve()

    print(f"The problem status is {prob.status}")
    print(f"The logistic loss is: {l.value}")
    print(f"The optimal value theta is: {theta.value}")
    print()

    print("Fitting robust logistic regression model...")
    theta_rob = cp.Variable(shape=d)
    l_rob = 0.0
    for i in range(n):
        y_i = y[i] # (1,)
        x_i = X[i] # (d,)
        z_i = -1 * y_i * theta_rob

        z_scaled = cp.abs(z_i * epsilon)
        u = z_i.T @ x_i + cp.sum(z_scaled)

        l_rob += cp.logistic(u)

    prob = cp.Problem(cp.Minimize(l_rob))
    print(f"The problem is dcp_compliant? {prob.is_dcp()}")
    prob.solve()

    print(f"The problem status is {prob.status}")
    print(f"The robust logistic loss is: {l_rob.value}")
    print(f"The optimal value theta is: {theta_rob.value}")
    print()

    def make_prediction(theta):
        return np.sign(X_test @ theta)

    def error_rate(y_pred):
        num_incorrect = np.sum(y_pred != y_test)
        num_total = y_test.size
        return float(num_incorrect) / num_total

    y_pred_log = make_prediction(theta.value) 
    error_rate_log = error_rate(y_pred_log)
    print(f"Logistic regression test error rate: {error_rate_log}")

    y_pred_robust_log = make_prediction(theta_rob.value)
    error_rate_rob_log = error_rate(y_pred_robust_log)
    print(f"Robut logistic regression test error rate: {error_rate_rob_log}")

if __name__ == "__main__":
    main()
