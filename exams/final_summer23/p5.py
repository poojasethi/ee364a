import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

from data.uav_design_data import *

def main():
    W_eng = cp.Variable(pos=True)
    W_bat = cp.Variable(pos=True)
    S = cp.Variable(pos=True)

    alpha = cp.Variable(K, pos=True)
    P = cp.Variable(K, pos=True)

    constraints = [
        W_eng_min <= W_eng,
        W_eng <= W_eng_max,
        W_bat_min <= W_bat,
        W_bat <= W_bat_max,
        S_min <= S,
        S <= S_max,
    ]

    inverse_power = 1 / 0.803

    for i in range(0, K):
        constraints.extend([
            # 0 <= alpha[i],  # not needed because enforce pos
            alpha[i] <= alpha_max,
            # 0 <= P[i], 
            P[i] <= (W_eng ** inverse_power) * (CP ** -inverse_power), 
            P[i] * D[i] * (V[i] ** -1) <= CE * W_bat,
        ])

        # For debugging only.
        # for constraint in constraints:
        #     if not constraint.is_dgp():
        #         breakpoint()

    def lift_coeff(a):
        return cL * a
    
    def drag_coeff(a):
        return cD1 + cD0 * (a ** 2)


    # Now add the constraints for the lift and drag forces. 
    additional_constraints = []
    for i in range(0, K):
        F_lift = 0.5 * rho * (V[i] ** 2) * lift_coeff(alpha[i]) * S
        W_wing = CW * (S ** 1.2)
        W = W_bat + W_eng + W_wing + W_pay[i] + W_base

        F_drag = 0.5 * rho * (V[i] ** 2) * drag_coeff(alpha[i]) * S
        T = P[i] * (V[i] ** -1)

        additional_constraints = [W <= F_lift, F_drag <= T]
        
        # For debugging only.
        # for constraint in additional_constraints:
        #     if not constraint.is_dgp():
        #         breakpoint()

    constraints.extend(additional_constraints)

    W_wing = CW * (S ** 1.2) 
    design_cost = 100 * W_eng + 45 * W_bat + 2 * W_wing
    mission_cost = cp.sum(cp.multiply(P, (V ** -1)) + 10 * alpha) 
    cost = design_cost + mission_cost

    prob = cp.Problem(cp.Minimize(cost), constraints)
    print(f"The problem is dgp_compliant? {prob.is_dgp()}")
    prob.solve(gp=True)

    print(f"The problem status is {prob.status}")
    print(f"The total optimal cost is {round(prob.value, 6)}")
    print()

    print("Optimal costs:")
    print(f"C_des={(100 * W_eng + 45 * W_bat + 2 * CW * (S ** 1.2)).value}")
    print(f"C_mis={cp.sum(cp.multiply(P, (V ** -1)) + 10 * alpha).value}")
    print()

    print("Design variables:")
    print(f"The optimal W_eng is {round(W_eng.value, 6)}")
    print(f"The optimal W_bat is {round(W_bat.value, 6)}")
    print(f"The optimal S is {round(S.value, 6)}")
    print()

    print("Mission variables:")
    print(f"The optimal alpha is {alpha.value}")
    print(f"The optimal power is {P.value}")
    print(f"")

    print("Are the relaxed constraints tight?")
    for i in range(0, K):
        print(f"Mission {i}")
        F_lift = 0.5 * rho * (V[i] ** 2) * lift_coeff(alpha[i]) * S
        W_wing = CW * (S ** 1.2)
        W = W_bat + W_eng + W_wing + W_pay[i] + W_base

        F_drag = 0.5 * rho * (V[i] ** 2) * drag_coeff(alpha[i]) * S
        T = P[i] * (V[i] ** -1) 

        print(f"\tW: {W.value}")
        print(f"\tF_lift: {F_lift.value}")

        print(f"\tF_drag: {F_drag.value}")
        print(f"\tT: {T.value}")

if __name__ == "__main__":
    main()
