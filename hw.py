import numpy as np
import matplotlib.pyplot as plt

# Given values
w = np.array([4, -5])
b = -2

# Create x range for plotting lines
x = np.linspace(-10, 10, 400)

# Decision boundary: 4x - 5y - 2 = 0 -> y = (4/5)x - 2/5
y_decision = (4/5) * x - (2/5)

# Orthogonal line to w (direction perpendicular to w is (5,4))
# Line through origin with direction (5,4): y = (4/5)x
y_ortho = (4/5) * x

# Plot setup
plt.figure(figsize=(7,7))

# (b) Orthogonal line
plt.plot(x, y_ortho, label='line orthogonal to weight vector')

# (c) Decision boundary
plt.plot(x, y_decision, label='decision boundary')

plt.plot([0, 4], [0, -5], label='weight vector')

plt.legend()
plt.grid()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()






























# from scipy.special import expit

# def relu(x):
#     return (x + np.abs(x))/2

# def cross_entropy_loss(yi, output):
#     return -yi * np.log(output)

# def relu_der(x):
#     return (x>0).astype(float)

# def sigmoid_grad(x):
#     return expit(x) * (1 - expit(x))

# def softmax(x):
#     return np.exp(x)/sum(np.exp(x))

# xi = np.array([6, 4, 7, 2, 1])
# yi = np.array([0, 1, 0])

# w1 = np.array([0, 4, -4, 5, 4])
# b_w1 = 1
# w2 = np.array([2, 2, -4, 3, 0])
# b_w2 = 1

# v1 = np.array([1, -2])
# b_v1 = -1
# v2 = np.array([2, 3])
# b_v2 = 0
# v3 = np.array([1, 2])
# b_v3 = 2

# xi_w1_pre = xi.dot(w1.T) + b_w1
# xi_w2_pre = xi.dot(w2.T) + b_w2

# xi_w1_post = relu(xi_w1_pre)
# xi_w2_post = relu(xi_w2_pre)

# w = np.array([xi_w1_post, xi_w2_post])

# w_v1 = expit(w.dot(v1.T) + b_v1)
# w_v2 = expit(w.dot(v2.T) + b_v2)
# w_v3 = expit(w.dot(v3.T) + b_v3)

# pre_output = np.array([w_v1, w_v2, w_v3])

# output = softmax(pre_output)
# print("output", output)

# print(cross_entropy_loss(yi, output))

# import numpy as np
# from scipy.special import expit

# def relu(x):
#     return (x + np.abs(x))/2

# def cross_entropy_loss(yi, output):
#     return -yi * np.log(output)

# def relu_der(x):
#     return (x>0).astype(float)

# def sigmoid_grad(x):
#     return expit(x) * (1 - expit(x))

# def softmax(x):
#     return np.exp(x)/sum(np.exp(x))

# # calculate gradients
# dpre_output = output - yi

# dz1 = dpre_output[0] * sigmoid_grad(w_v1)
# dz2 = dpre_output[1] * sigmoid_grad(w_v2)
# dz3 = dpre_output[2] * sigmoid_grad(w_v3)

# dz = np.array([dz1, dz2, dz3])
# dv1 = np.outer(w, dz1)
# dv2 = np.outer(w, dz2)
# dv3 = np.outer(w, dz3)

# dh = (dz1 * v1) + (dz2 * v2) + (dz3 * v3)
# du = dh * relu_der(np.array([xi_w1_pre, xi_w2_pre]))

# dw1 = xi * du[0]
# dw2 = xi * du[1]


# print("dw1:", dw1)
# print("dw2:", dw2)
# print("dv1:", dv1)
# print("dv2:", dv2)
# print("dv3:", dv3)