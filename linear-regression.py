def hypothesis_function(theta, x_i):
    return sum([theta_j * x_i_j for theta_j, x_i_j in zip(theta, x_i)])


def cost_function(theta, x, y):
    return 0.5 * sum([pow(hypothesis_function(theta, x_i) - y_i, 2) for x_i, y_i in zip(x, y)])


def cost_function_derivative(theta, x, y, j):
    return sum([(hypothesis_function(theta, x_i) - y_i) * x_i[j] for x_i, y_i in zip(x, y)])


learning_rate = 0.01
training_set = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5]]

m = len(training_set)
n = len(training_set[0]) - 1

x = [example[:-1] for example in training_set]
y = [example[-1] for example in training_set]

theta = [0 for _ in range(n)]


for epoch in range(100):
    print(cost_function(theta, x, y))
    next_theta = [
        theta_j - learning_rate * cost_function_derivative(theta, x, y, j)
        for j, theta_j in enumerate(theta)
    ]
    theta = next_theta

print(theta)