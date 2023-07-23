import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from typing import Any, Tuple

np.random.seed(0)


def barrier_method(A, c, b, x_0, t_0=1, epsilon=1e-3, mu=10) -> Tuple[Any, Any, Any, Any]:
    """Solves the standard form LP using the barrier method.

    Returns:
        x_star: primal optimal point
        nu_star: dual optimal point (equality constraints)
        lamb_star: dual optimal point (inequality constraints)
        history: 2 x k matrix (k is the total number of centering steps)
            first row contains the number of Newton steps required for each centering step
            second row contains shows the duality gap at the end of each centering step
    """
    m, n = A.shape
    x, t = x_0, t_0

    newton_steps_per_iteration = []
    duality_gaps_per_iteration = []

    duality_gap = float(n) / t

    x_star = None
    nu_star = None
    lamb_star = None

    while True:
        x_star, nu_star, num_newton_steps = infeasible_start_newton_method(A, t * c, b, x)
        x = x_star
        nu = nu_star
        lamb = 1.0 / (-t * x) if x is not None else None

        # Centering step failed
        if x_star is None:
            break

        duality_gap = float(n) / t
        duality_gaps_per_iteration.append(duality_gap)
        newton_steps_per_iteration.append(num_newton_steps)

        if duality_gap < epsilon:
            break

        t = mu * t

    x_star = x
    nu_star = nu
    lamb_star = lamb

    history = np.vstack((np.array(newton_steps_per_iteration), np.array(duality_gaps_per_iteration)))
    return (x_star, nu_star, lamb_star, history)


def infeasible_start_newton_method(A, c, b, x_0, max_iterations=50, alpha=0.1, beta=0.5):
    """Runs infeasible start Newton method from homework 7."""
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

    # print(f"Number of Newton steps: {num_newton_steps}")
    # print(f"Norm of residual: {res_current_norm}")
    # print(f"Found solution? {res_current_norm <= 1e-6}")

    x_star = x if res_current_norm <= 1e-6 else None
    nu_star = nu if res_current_norm <= 1e-6 else None
    return x_star, nu_star, num_newton_steps


def main():
    print("Test feasible instance")
    m = 100
    n = 500

    # Generate matrix A. Re-sample until we are sure it is full rank.
    A = np.random.rand(m, n)
    while np.linalg.matrix_rank(A) != m:
        A = np.random.rand(m, n)

    x_0 = np.random.rand(n) + 0.1
    b = A @ x_0

    c = np.random.uniform(low=-1.0, high=1.0, size=n)

    x_star = None
    for mu in [2, 15, 50, 150]:
        print(f"Running barrier method for {mu=}")
        x_star, nu_star, lamb_star, history = barrier_method(A, c, b, x_0, mu=mu)
        # print(f"x_star: {x_star}")
        # print(f"nu_star: {nu_star}")
        # print(f"lamb_star: {lamb_star}")
        print()
        plt.step(np.cumsum(history[0, :]), history[1, :], where="post", label=r"$\mu$={mu}".format(mu=mu))

    plt.yscale("log")
    plt.legend()
    plt.xticks(range(0, 110, 10))
    plt.show()

    print("Solving with CVXPY...")
    x = cp.Variable(n)
    constraints = [A @ x == b, x >= 0]
    problem = cp.Problem(cp.Minimize(c.T @ x), constraints)
    problem.solve()
    print(f"Problem status: {problem.status}")

    print(f"CVXPY optimal value: {problem.value}")
    print(f"Barrier method optimal value: {c.T @ x_star}")
    print(f"l2 norm between CVXPY x_star and barrier method x_star: {np.linalg.norm(x.value - x_star)}")
    print("*" * 50)
    ###############################################################
    print("Test unbounded below.")
    m = 3
    n = 4
    A = np.hstack([np.eye(m), np.zeros((m, 1))])
    x_0 = np.ones(n)
    b = A @ x_0

    c = -np.ones(n)

    x_star, nu_star, lamb_star, history = barrier_method(A, c, b, x_0)
    print("Solving with CVXPY...")
    x = cp.Variable(n)
    problem = cp.Problem(cp.Minimize(c.T @ x), constraints=[A @ x == b, x >= 0])
    problem.solve()
    print(f"Problem status: {problem.status}")
    print(f"CVXPY optimal value: {problem.value}")
    print(f"Barrier method optimal x_star: {x_star}")

    print("*" * 50)
    ###############################################################
    print("Test infeasible.")
    m = 3
    n = 4
    A = np.ones((m, n))
    b = -np.ones(m)
    c = -np.ones(n)

    x_0 = np.ones(n)

    x_star, nu_star, lamb_star, history = barrier_method(A, c, b, x_0)

    problem = cp.Problem(cp.Minimize(c.T @ x), constraints=[A @ x == b, x >= 0])
    problem.solve()
    print(f"Problem status: {problem.status}")
    print(f"CVXPY optimal value: {problem.value}")
    print(f"Barrier method optimal x_star: {x_star}")


if __name__ == "__main__":
    main()
