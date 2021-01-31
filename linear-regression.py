import math


def hypothesis_function(theta, x_i):
    return sum([theta_j * x_i_j for theta_j, x_i_j in zip(theta, x_i)])


def cost_function(theta, x, y):
    return 0.5 * sum([pow(hypothesis_function(theta, x_i) - y_i, 2) for x_i, y_i in zip(x, y)])


def cost_function_derivative(theta, x, y, j):
    return sum([(hypothesis_function(theta, x_i) - y_i) * x_i[j] for x_i, y_i in zip(x, y)])


def is_theta_converged(theta, next_theta):
    for theta_j, next_theta_j in zip(theta, next_theta):
        if not math.isclose(theta_j, next_theta_j):
            return False
    return True


learning_rate = 0.01
training_set = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5]]

m = len(training_set)  # 데이터 개수
n = len(training_set[0]) - 1  # input 차원, input feature 개수

X = [[1] + example[:-1] for example in training_set]  # X_i는 vector
y = [example[-1] for example in training_set]  # y_i는 value

theta = [0 for _ in range(n + 1)]


while True:
    next_theta = [
        theta_j - learning_rate * cost_function_derivative(theta, X, y, j)
        for j, theta_j in enumerate(theta)
    ]

    if is_theta_converged(theta, next_theta):
        break

    theta = next_theta

print(hypothesis_function(theta, [1, 2, 2, 2]))

# Normal Equation