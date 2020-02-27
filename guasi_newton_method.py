import numpy as np
import matplotlib.pyplot as plt


X = np.array([np.ones(100), np.random.rand(100)])
y = np.dot([4, 3], X) + np.random.rand(100)
plt.scatter(X[1, :], y)


# 拟牛顿法的实现
# 损失量
def msc_loss(theta, X, y, m):
    loss = (1/(2*m))*np.sum(np.square(np.dot(theta, X)-y))
    return loss


# 求当前theta下的梯度
def grad(theta, X, y, m):
    grad = (1/m)*np.dot(np.dot(theta, X)-y, X.T)
    return grad


# 输出关键信息
def print_key_info(theta, loss, round):
    print("In round: ", round)
    print("Theta = ", theta)
    print("Loss = ", loss, "\n")


# 拟牛顿法回归
def guasi_newton(theta, X, y, iter_num):
    n = X.shape[0]
    m = y.size

    # theta、损失量、D（使用相邻两次梯度替代的Hessian矩阵的逆矩阵）的所有记录
    theta_history = np.zeros((iter_num+1, n))
    loss_history = np.zeros(iter_num+1)
    D_history = np.zeros((iter_num+1, n, n))

    # 给theta、损失量和D矩阵赋初始值
    theta_history[0, :] = theta
    loss_history[0] = msc_loss(theta, X, y, m)
    # 设D0为单位矩阵
    D_history[0] = np.eye(n)
    print_key_info(theta, loss_history[0], 0)

    for i in range(1, iter_num+1):

        # 使用DFP算法完成拟牛顿法
        # 先求出delta_D[k]，delta_D[k]，求delta_D[k]之前，需要先求出theta[k+1]和g[k+1]
        # D[k+1] = D[k] + delta_D[k]
        # S[k] = theta[k+1] - theta[k]
        # Y[k] = g[k+1] - g[k]
        g = grad(theta_history[i-1], X, y, m)
        D = D_history[i-1]

        theta_prev = theta_history[i-1]
        theta = theta_prev - np.dot(D, g)
        theta_history[i, :] = theta
        loss_history[i] = msc_loss(theta, X, y, m)

        Sk = theta - theta_prev
        Yk = grad(theta, X, y, m) - g

        delta_D = np.dot(Sk, Sk.T)/np.dot(Sk.T, Sk) - np.dot(D, Yk).dot(np.dot(Yk.T, D.T))/np.dot(Yk.T, D).dot(Yk)
        D_history[i] = D + delta_D
        # if i % 100 == 0:
        print_key_info(theta, loss_history[i], i)

    return theta_history, loss_history, D_history


theta_init = np.random.rand(2)
iter_num = 10
theta_history, loss_history, D_history = guasi_newton(theta_init, X, y, iter_num)

print("The Final: ")
print("Theta = ", theta_history[-1])
print("Loss = ", loss_history[-1])

# from Newton_method import newton_method
# theta_f, theta_history_f, loss_history_f = newton_method(X, y, theta_init, iter_num)
#
# print("Newton Method: ")
# print("Theta = ", theta_history_f[-1])
# print("Loss = ", loss_history_f[-1])
