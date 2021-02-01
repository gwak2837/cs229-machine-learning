import math
import numpy as np
from decimal import Decimal


def perceptron_function(z):
    if z >= 0:
        return 1
    else:
        return 0


def hypothesis_function(theta, X_i):
    return perceptron_function(sum([theta_j * X_i_j for theta_j, X_i_j in zip(theta, X_i)]))


def update_theta(theta, X, y, j):
    return sum([(y_i - hypothesis_function(theta, X_i)) * X_i[j] for X_i, y_i in zip(X, y)])


def is_theta_converged(theta, next_theta):
    for theta_j, next_theta_j in zip(theta, next_theta):
        if not math.isclose(theta_j, next_theta_j):
            return False
    return True


training_set = np.genfromtxt("binary-logistic-regression-training-data.txt", delimiter=",").tolist()

m = len(training_set)  # 데이터 개수
n = len(training_set[0]) - 1  # input 차원, input feature 개수

X = [[1] + example[:-1] for example in training_set]  # X_i는 vector
y = [int(example[-1]) for example in training_set]  # y_i는 value

learning_rate = [200, 0.01, 0.01]
theta = [-25, 0.2, 0.2]


while True:
    print(theta[0], "\t", theta[1], "\t", theta[2])

    next_theta = [
        theta_j + learning_rate[j] * update_theta(theta, X, y, j) for j, theta_j in enumerate(theta)
    ]

    if is_theta_converged(theta, next_theta):
        break

    theta = next_theta

print(theta)


# Newton's Method