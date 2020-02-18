import numpy as np
import matplotlib.pyplot as plt


X = np.array([np.ones(100), np.random.rand(100)])
y = np.dot([4, 3], X) + np.random.rand(100)
plt.scatter(X[1, :], y)

alpha = 0.01
alter_num = 1000
theta_init = np.random.randn(2)
precise = 0.1


def newton_method(X, y, theta, alter_num):
    m = len(y)
    print("\nInit theta = {}".format(theta))
    theta_history = np.array(np.zeros((alter_num, 2)))
    loss_history = np.array(np.zeros(alter_num))

    for i in range(alter_num):

        y_pred = np.dot(theta, X)

        theta = theta - np.dot(np.dot(y_pred - y, X.T),
                               np.linalg.inv(np.dot(X, X.T)))
        loss = (1/(2 * m)) * np.sum(np.square(y_pred - y))

        theta_history[i, :] = theta
        loss_history[i] = loss

        if i % 100 == 0:
            print("In {}: ".format(i))
            print("theta = {}".format(theta))
            print("loss = {}\n".format(loss))

    return theta, theta_history, loss_history


theta_n, theta_history_n, loss_history_n = newton_method(X, y, theta_init, alter_num)

print("Final theta improved by Newton Method = {}".format(theta_n))
print("Final loss improved by Newton Method = {}".format(loss_history_n[-1]))
# plt.plot(loss_history)
# plt.plot(theta_history)
x_plot = np.linspace(0, 1)
y_plot_n = theta_n[0] + theta_n[1] * x_plot

plt.scatter(X[1, :], y)
plt.plot(x_plot, y_plot_n, color='red')

plt.show()

