import numpy as np
import matplotlib.pyplot as plt

X = np.array([np.ones(100), np.random.rand(100)])
y = np.dot([4, 3], X) + np.random.rand(100)
plt.scatter(X[1, :], y)

alpha = 0.05
alter_num = 1000
theta_init = np.random.randn(2)


def gradient_descent(X, y, theta, alpha, alter_num):
    m = len(y)
    print("\nInit theta = {}".format(theta))
    theta_history = np.array(np.zeros((alter_num, 2)))
    loss_history = np.array(np.zeros(alter_num))

    for i in range(alter_num):

        y_pred = np.dot(theta, X)

        theta = theta - (alpha / m) * np.dot(y_pred - y, X.T)
        loss = (1/(2 * m)) * np.sum(np.square(y_pred - y))

        theta_history[i, :] = theta
        loss_history[i] = loss

        if i % 100 == 0:
            print("In {}: ". format(i))
            print("theta = {}". format(theta))
            print("loss = {}\n". format(loss))
    return theta, theta_history, loss_history


theta, theta_history, loss_history = gradient_descent(X, y, theta_init, alpha, alter_num)
print("Final theta = {}".format(theta))
# plt.plot(loss_history)
# plt.plot(theta_history)
x_plot = np.linspace(0, 1)
y_plot = theta[0] + theta[1] * x_plot

plt.plot(x_plot, y_plot)
plt.show()

