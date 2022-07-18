from coefficients import coeff_1 as c_1
from coefficients import coeff_2 as c_2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton

max_radius = 0.92                               # radius of the brb
max_current = 7000                              # maximum current value for the torroidal field coil
num_points = 10000
r = np.linspace(0, max_radius, num_points)
i = np.linspace(0, max_current, num_points)

c1_abs = lambda r: c_1(r, i)                    # lambda functions to get the coefficients 1 and 2 as a function of the radius
c2_abs = lambda r: c_2(r, i)

zero_c1 = newton(c1_abs, i * 10**-6)            # creates an array of radius values for which the coefficients 1 and 2 are zero
zero_c2 = newton(c2_abs, i * 10**-6)            # zeroes are computed using the Newton-Raphson method

# Plotting the radius values at which coeffiencts 1 and 2 are zero for different current values
plt.figure()

plt.plot(zero_c1, i, color = 'r')
plt.plot(zero_c2, i, color = 'b')
plt.xlabel('Radius')
plt.ylabel('Current')
plt.title('Current vs Radius (for coefficients 1, 2 = 0)')

plt.legend()
plt.show()