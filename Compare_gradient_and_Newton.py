import numpy as np
import matplotlib.pyplot as plt

X = np.array([np.ones(100), np.random.rand(100)])
y = np.dot([4, 3], X) + np.random.rand(100)
plt.scatter(X[1, :], y)

alpha = 0.05
alter_num = 1000
theta_init = np.random.randn(2)


def gradient_descent(X, y, theta, alpha, m, y_pred):
    return theta - (alpha / m) * np.dot(y_pred - y, X.T)


def newton_method(X, y, theta, alpha, m, y_pred):
    return theta - np.dot(np.dot(y_pred - y, X.T),
                               np.linalg.inv(np.dot(X, X.T)))

def frame_work(X, y, theta, alpha, alter_num, keyfunc):
    m = len(y)
    print("\nInit theta = {}".format(theta))
    theta_history = np.array(np.zeros((alter_num, 2)))
    loss_history = np.array(np.zeros(alter_num))

    for i in range(alter_num):

        y_pred = np.dot(theta, X)

        theta = keyfunc(X, y, theta, alpha, m, y_pred)
        loss = (1/(2 * m)) * np.sum(np.square(y_pred - y))

        theta_history[i, :] = theta
        loss_history[i] = loss

        if i % 100 == 0:
            print("In {}: ". format(i))
            print("theta = {}". format(theta))
            print("loss = {}\n". format(loss))
    return theta, theta_history, loss_history


theta_g, theta_history_g, loss_history_g = frame_work(X, y, theta_init, alpha, alter_num, gradient_descent)
theta_n, theta_history_n, loss_history_n = frame_work(X, y, theta_init, alpha, alter_num, newton_method)

print("Final theta calculated by gradient descent = {}".format(theta_g))
print("Final loss calculated by gradient descent = {}".format(loss_history_g[-1]))

print("Final theta improved by Newton Method = {}".format(theta_n))
print("Final loss improved by Newton Method = {}".format(loss_history_n[-1]))
# plt.plot(loss_history)
# plt.plot(theta_history)
x_plot = np.linspace(0, 1)
y_plot_n = theta_n[0] + theta_n[1] * x_plot
y_plot_g = theta_g[0] + theta_g[1] * x_plot

plt.scatter(X[1, :], y)
plt.plot(x_plot, y_plot_n, color='red')
plt.plot(x_plot, y_plot_g, color='green')

plt.show()

