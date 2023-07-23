import numpy as np
import matplotlib.pyplot as plt
import itertools

np.random.seed(0)


def infeasible_start_newton_method(A, c, b, x_0, max_iterations=50, alpha=0.1, beta=0.5):
    m, n = A.shape

    # Check dimensions of input
    assert m < n, f"m must be less than n, but got {m=} and {n=}"
    assert c.shape == (n,), f"c should be of size n, but got {c.shape=}"
    assert b.shape == (m,), f"b should be of size m, but got {b.shape=}"
    assert x_0.shape == (n,), f"x_0 should be of size n, but got {x_0.shape=}"

    # Check that x_0 is positive.
    assert np.all(x_0 > 0), f"All values of x_0 should be positive"

    num_newton_steps = 0

    x = x_0
    nu = np.zeros(m)
    all_residual_norms = []

    # See equation 10.21 on pg. 533 of book for equations.
    def residual(x, nu, A):
        r_dual = (c - x**-1) + A.T @ nu  # vector of shape (n, )
        r_primal = A @ x - b  # vector of shape (m, )
        return -1 * np.hstack((r_dual, r_primal))

    res_current = residual(x, nu, A)
    res_current_norm = np.linalg.norm(res_current)

    # Method from slide 11-10
    while res_current_norm > 1e-6 and num_newton_steps < max_iterations:
        # Compute primal and dual steps directly from KKT system.
        # (Could also use block elimination.)
        H = np.diag(x**-2)
        M = np.hstack((np.vstack((H, A)), np.vstack((A.T, np.zeros((m, m))))))
        assert M.shape == (n + m, n + m)

        delta = None
        try:
            delta = np.linalg.solve(M, res_current)
        except Exception as e:
            print(e)
            return None, None, 0

        x_delta = delta[:n]
        nu_delta = delta[-m:]

        # Perform backtracking line search
        t = 1
        while np.any(x + t * x_delta < 0):
            t = beta * t

        while np.linalg.norm(residual(x + t * x_delta, nu + t * nu_delta, A)) > (1 - alpha * t) * np.linalg.norm(
            residual(x, nu, A)
        ):
            t = beta * t

        # Perform update
        x = x + t * x_delta
        nu = nu + t * nu_delta
        res_current = residual(x, nu, A)
        res_current_norm = np.linalg.norm(res_current)
        all_residual_norms.append(res_current_norm)

        num_newton_steps += 1

    print(f"Number of Newton steps: {num_newton_steps}")
    print(f"Norm of residual: {res_current_norm}")
    print(f"Found solution? {res_current_norm <= 1e-6}")

    # Plot norms of residuals vs. number of iterations
    plt.yscale("log")
    plt.plot(range(1, num_newton_steps + 1), all_residual_norms)
    plt.xticks(range(0, num_newton_steps + 1))
    plt.xlabel("Iteration")
    plt.ylabel("Norm of residual")
    plt.title(f"{alpha=}, {beta=}")
    # plt.show()

    x_star = x if res_current_norm <= 1e-6 else None
    nu_star = nu if res_current_norm <= 1e-6 else None
    return x_star, nu_star, num_newton_steps


def main():
    print("Test feasible instances.")
    m = 100
    n = 500

    # Generate matrix A. Re-sample until we are sure it is full rank.
    A = np.random.rand(m, n)
    while np.linalg.matrix_rank(A) != m:
        A = np.random.rand(m, n)

    p = np.random.rand(n)
    b = A @ p

    c = np.random.uniform(low=-1.0, high=1.0, size=n)

    x_0 = np.ones(n)

    alphas = [0.1, 0.5]
    betas = [0.1, 0.5]

    for alpha, beta in itertools.product(alphas, betas):
        print(f"{alpha=}")
        print(f"{beta=}")
        x_star, nu_star, num_newton_steps = infeasible_start_newton_method(A, c, b, x_0, alpha=alpha, beta=beta)
        print()

    print("*" * 50)
    ###############################################################
    print("Test unbounded below.")
    m = 3
    n = 4
    A = np.hstack([np.eye(m), np.zeros((m, 1))])
    b = np.ones(m)
    c = -np.ones(n)

    x_0 = np.ones(n)

    x_star, nu_star, num_newton_steps = infeasible_start_newton_method(A, c, b, x_0)
    print("*" * 50)
    ###############################################################
    # Test infeasible.
    A = None  # A has negative entries in each row

    m = 3
    n = 4
    A = np.hstack([np.eye(m), np.zeros((m, 1))])
    b = -np.ones(m)
    c = -np.ones(n)

    x_0 = np.ones(n)
    x_star, nu_star, num_newton_steps = infeasible_start_newton_method(A, c, b, x_0)


if __name__ == "__main__":
    main()
