import numpy as np
import cvxpy as cp

K = 0.5

p = np.array([4, 2, 2, 1])
d = np.array([20, 5, 10, 15])
s = np.array([30, 10, 5, 0])
d_tilde = np.array([10, 25, 5, 15])
s_tilde = np.array([5, 20, 15, 20])


def main():
    t = cp.Variable(4)
    B = cp.Variable(shape=(4, 4))
    B_tilde = cp.Variable(shape=(4, 4))
    ones = np.ones(4)

    cost = K * cp.norm(t, 1) + p.T @ (B.T @ ones + B_tilde.T @ ones)

    s_plus = s - t
    s_tilde_plus = s_tilde + t

    constraints = [s_plus >= 0, s_tilde_plus >= 0]

    B_constraints = [
        B @ ones == d,
        B.T @ ones == s_plus,
        B >= 0,
        # Sparsity constraints
        B[0][1] == 0,  # type O cannot accept A
        B[0][2] == 0,  # type O cannot accept B
        B[0][3] == 0,  # type O cannot accept AB
        B[1][2] == 0,  # type A cannot accept B
        B[1][3] == 0,  # type A cannot accept AB
        B[2][1] == 0,  # type B cannot accept A
        B[2][3] == 0,  # type B cannot accept AB
    ]

    B_tilde_constraints = [
        B_tilde @ ones == d_tilde,
        B_tilde.T @ ones == s_tilde_plus,
        B_tilde >= 0,
        # Sparsity constraints
        B_tilde[0][1] == 0,  # type O cannot accept A
        B_tilde[0][2] == 0,  # type O cannot accept B
        B_tilde[0][3] == 0,  # type O cannot accept AB
        B_tilde[1][2] == 0,  # type A cannot accept B
        B_tilde[1][3] == 0,  # type A cannot accept AB
        B_tilde[2][1] == 0,  # type B cannot accept A
        B_tilde[2][3] == 0,  # type B cannot accept AB
    ]

    constraints.extend(B_constraints)
    constraints.extend(B_tilde_constraints)

    problem = cp.Problem(cp.Minimize(cost), constraints=constraints)
    problem.solve()

    print(f"Problem status: {problem.status}")

    print(f"Optimal cost: {problem.value}")
    print(f"Optimal shipment vector: t={t.value}")
    print(f"Optimal policy B:\n{B.value}")
    print(f"Optimal policy B_tilde:\n{B_tilde.value}")
    print()

    print("Shipments no longer allowed:")
    constraints.extend([t == np.zeros(4)])
    problem = cp.Problem(cp.Minimize(cost), constraints=constraints)
    problem.solve()
    print(f"Problem status: {problem.status}")


if __name__ == "__main__":
    main()
