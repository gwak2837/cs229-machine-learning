import math


def is_theta_converged(theta, next_theta):
    for theta_j, next_theta_j in zip(theta, next_theta):
        if not math.isclose(theta_j, next_theta_j):
            return False
    return True