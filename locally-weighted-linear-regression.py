import math
from commons import is_theta_converged

tou = 1


def weight_function(X_i, testing_X, tou):
    return math.exp(
        -pow(sum([abs(X_i_j - testing_X_j) for X_i_j, testing_X_j in zip(X_i, testing_X)]), 2)
        / (2 * tou * tou)
    )


def hypothesis_function(theta, X_i):
    return sum([theta_j * X_i_j for theta_j, X_i_j in zip(theta, X_i)])


def cost_function(theta, X, y):
    return 0.5 * sum(
        [
            pow(hypothesis_function(theta, X_i) - y_i, 2) * weight_function(X_i, X, tou)
            for X_i, y_i in zip(X, y)
        ]
    )


def cost_function_derivative(theta, X, y, j, testing_X):
    return sum(
        [
            (hypothesis_function(theta, X_i) - y_i) * X_i[j] * weight_function(X_i, testing_X, tou)
            for X_i, y_i in zip(X, y)
        ]
    )


learning_rate = 0.002
training_set = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5]]

m = len(training_set)  # 데이터 개수
n = len(training_set[0]) - 1  # input 차원, input feature 개수

X = [[1] + example[:-1] for example in training_set]  # X_i는 vector
y = [example[-1] for example in training_set]  # y_i는 scalar

theta = [0 for _ in range(n + 1)]

testing_X = [1.5, 1.5, 1.5]

while True:
    print(theta)
    next_theta = [
        theta_j - learning_rate * cost_function_derivative(theta, X, y, j, testing_X)
        for j, theta_j in enumerate(theta)
    ]

    if is_theta_converged(theta, next_theta):
        break

    theta = next_theta

print(theta)
