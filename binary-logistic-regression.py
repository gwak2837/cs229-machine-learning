import math
import numpy as np
from decimal import Decimal

k = 0.001


def logistic_function(z):
    return 1 / (1 + math.exp(-k * z))


def logistic_function_derivative(z):
    return k * logistic_function(z) * (1 - logistic_function(z))


def hypothesis_function(theta, X_i):
    return logistic_function(sum([theta_j * X_i_j for theta_j, X_i_j in zip(theta, X_i)]))


def log_likelyhood_function(theta, X, y):
    return sum(
        [
            math.log(hypothesis_function(theta, X_i))
            if y_i == 1
            else math.log(1 - hypothesis_function(theta, X_i))
            for X_i, y_i in zip(X, y)
        ]
    )


def log_likelyhood_function_derivative(theta, X, y, j):
    return k * sum([(y_i - hypothesis_function(theta, X_i)) * X_i[j] for X_i, y_i in zip(X, y)])


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

learning_rate = [i * (1 / k) for i in [200, 0.01, 0.01]]
theta = [0 for _ in range(n + 1)]


while True:
    print(theta[0], "\t", theta[1], "\t", theta[2])
    print(log_likelyhood_function(theta, X, y))
    next_theta = [
        theta_j + learning_rate[j] * log_likelyhood_function_derivative(theta, X, y, j)
        for j, theta_j in enumerate(theta)
    ]

    if is_theta_converged(theta, next_theta):
        break

    theta = next_theta

print(theta)


# Newton's Method