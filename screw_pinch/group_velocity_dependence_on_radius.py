from coefficients import coeff_1 as c_1
from coefficients import coeff_2 as c_2
from dispersion_relation import get_omega
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative as diff
from scipy.optimize import newton

# Defining the physical quantites 
max_radius = 0.92
r = np.linspace(0, max_radius, 10000)
k = 1000
i = 4000
dk = 10**-6

def get_group_velocity(k, r, i):
    """
    Get the group velocity for different values of wavenumber using the lambda fuction

    params:
    k = wavenumber values
    r = radius values
    i = current values
    
    """
    #xsq = lambda x: x**2
    x = lambda k: get_omega(k, r, i)
    return(diff(x, k, dx=dk))

c1_abs = lambda r: c_1(r, i)                # lambda functions to get the coefficients 1 and 2 as a function of radius
c2_abs = lambda r: c_2(r, i)

if i == 0:
    zero_c1 = 0
else:
    zero_c1 = newton(c1_abs, i * 10**-6)    # creates an array of radius values for which the coefficients 1 and 2 are zero
                                            # zeroes are computed using the Newton-Raphson method
if i == 0:
    zero_c2 = 0
else:    
    zero_c2 = newton(c2_abs, i * 10**-6)


# Plotting the group velocity at different values of radius along with the zero point for coefficients 1 and 2
plt.figure()

plt.plot(r, get_group_velocity(k, r, i).real, label = 'Real part of the group velocity')
plt.plot(zero_c1, 0, marker='o')
plt.plot(zero_c2, 0, marker='o')
plt.hlines(0, 0, max_radius, colors='black')
plt.ylim(-25,25)
plt.xlabel('Radius')
plt.ylabel('Group Velocity')
plt.title("TF Current = {current} A".format(current=i))

plt.legend()
plt.show()