import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

def main():
    T = cp.Variable(pos=True)
    r = cp.Variable(pos=True)
    w = cp.Variable(pos=True)

    a_1, a_2, a_3, a_4 = 0.2, 0.1, 0.3, 1
    C_max = 60
    T_min, T_max = 10, 40
    r_min, r_max = 35, 80
    w_min, w_max = 3, 4
   
    constraints = [
       T_min <= T,
       T <= T_max, 
       r_min <= r,
       r <= r_max,
       w_min <= w,
       w <= w_max,
       w <= 0.1 * r,
       a_1 * T * r * w ** -1 + a_2 * r + a_3 * r * w <= C_max,
    ]

    heat_flow = a_4 * T * (r ** 2)
    prob = cp.Problem(cp.Maximize(heat_flow), constraints)
    print(f"The problem is dgp_compliant? {prob.is_dgp()}")
    prob.solve(gp=True)

    print(f"The problem status is {prob.status}")
    print(f"The optimal heat flow is {round(prob.value, 6)}")
    print(f"The optimal cost is {round((a_1 * T * r * w ** -1 + a_2 * r + a_3 * r * w).value , 6)}")

    print(f"The optimal T is {round(T.value, 6)}")
    print(f"The optimal r is {round(r.value, 6)}")
    print(f"The optimal w is {round(w.value, 6)}")
    print("*" * 100)


if __name__ == "__main__":
    main()
