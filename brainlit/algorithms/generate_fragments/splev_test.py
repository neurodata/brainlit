import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from spline_fxns import splev_degreecontrol

# theta = np.linspace(0, 2*np.pi)
# y = np.sin(theta)
# tck, u = splprep([y])
# new_points = splev(u, tck)
# x = u
# xi = tck[0]
# cs = tck[1]
# p = tck[2]
# print("x aka u")
# print(x)
# print("xi aka the knots")
# print(xi)
# print("cs aka coefficients aka control points")
# print(cs)

# print()

# plt.plot(theta, y, "ro")
# plt.plot(u*2*np.pi, new_points[0], "r-")
# plt.show()

x = np.linspace(-1, 1)
pts = np.array([[-1, 1], [-0.5, 1], [0, 1], [0.5, 1], [1, 1]])
tck, u = splprep(pts.T)
cs = tck[1]
cs = np.transpose(cs, (1, 0))
print(cs)
# newpoints = splev(np.linspace(0, 1), tck)
# val = splev_degreecontrol(np.linspace(0, 1), tck)
# print(val)
# plt.plot(newpoints[0], newpoints[1])
# plt.show()
# p = 0
# n = len(cs)
# xi_L = p + n + 1
# xi = np.linspace(0, 1, xi_L)
# tck = tuple([xi, cs, p])
# b = splev_degreecontrol(x, tck)
# print(b)
