from re import X
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative as diff
from scipy.optimize import newton

# Setting up the physical quatities that are constant

b_helmholtz = 0.005                        # The magnetic field generated due to the Helmholtz coil
mu_0 = 4 * np.pi * 10**(-7)
max_radius = 0.92                          # Radius of the brb
max_current = 7000                         # Maximum current value for the torroidal field coil
r = np.linspace(0, max_radius, 100000)
i = 5000
dr = 10**-6                                # Differential step for differentiation done while calculating values for different fuctions
bounds = [(0, max_radius)]

def b_tf(r, i):
    """
    Get the torroidal field at various r and i values

    params:
    r = radius values
    i = current values

    """
    if isinstance(r, float):
        if r==0:
            return np.inf
        else:
            return (mu_0 * i)/(2 * np.pi * r)
    else:
        return np.where(r==0, np.inf, (mu_0 * i)/(2 * np.pi * r))

def beta(r):
    """
    Get the plasma confinement factor at various r values

    params:
    r = radius

    """
    return (((1 + np.cos((np.pi * r)/max_radius)) * 0.02) + 0.01)

def b_z(r):
    """
    Get the residual magnetic field after the interaction
    between plasma and background magnetic field at various r values

    params:
    r = radius

    """
    return ((b_helmholtz)/(np.sqrt(1 + beta(r))))

def p_0(r):
    """
    Get the density of the plasma at various r values

    params:
    r = radius values

    """
    return ((b_helmholtz**2 - b_z(r)**2)/(2 * mu_0))

# Calculation for getting the coefficient of radial displacement vector(\xi\)
def coeff_1(r, i):
    term_1 = (-5/3) * b_helmholtz**2 + 2 * r * diff(b_tf, r, args=[i], dx=dr) * (2 * b_tf(r, i) + r * diff(b_tf, r, args=[i], dx=dr))
    term_2 = (-1/3) * b_z(r) * (b_z(r) - 2 * r * diff(b_z, r, dx=dr))
    term_3 = 2 * r**2 * b_tf(r, i) * diff(b_tf, r, n=2, args=[i], dx=dr)

    return ((1/(2 * mu_0 * r**2 * p_0(r))) * (term_1 + term_2 + term_3))

# Calculation for getting the coefficient of first order partial differential of the radial displacement vector(\xi\)
def coeff_2(r,i):
    term_1 = (5/3) * b_helmholtz**2 + 4 * b_tf(r,i)**2 + 6 * r * b_tf(r,i) * diff(b_tf, r, args=[i], dx=dr)
    term_2 = (1/3) * b_z(r) * (b_z(r) + 2 * r * diff(b_z, r, dx=dr))

    return ((1/(2 * mu_0 * r * p_0(r))) * (term_1 + term_2))

# Calculation for getting the coefficient of second order partial differential of the radial displacement vector(\xi\)
def coeff_3(r,i):
    term1 = (5/3) * b_helmholtz**2 + 2 * b_tf(r,i)**2 + (b_z(r)**2)/3

    return ((1/(2 * mu_0 * p_0(r))) * term1)


def plot_coefficents(r, i):
    # Plotting the coefficients over range of radius values for a constant value of current
    
    c1_abs = lambda r: coeff_1(r, i)
    c2_abs = lambda r: coeff_2(r, i)
    

    zero_c1 = newton(c1_abs, i * 10**-6)
    print(zero_c1)

    zero_c2 = newton(c2_abs, i * 10**-6)
    print(zero_c2)

    plt.plot(r, coeff_1(r,i), color = 'r', label = 'coefficient 1')
    plt.plot(r, coeff_2(r,i), color = 'g', label = 'coefficient 2')
    plt.plot(r, coeff_3(r,i), color = 'b', label = 'coefficient 3')
    plt.axhline(0, color = 'black')
    plt.plot(zero_c1, 0, marker='o')
    plt.plot(zero_c2, 0, marker='o')
    plt.ylim(-500, 500)
    plt.xlabel('Radius')
    plt.ylabel('Coefficients')
    plt.title("TF Current = {current} A".format(current=i))

    plt.legend()

    plt.show()

if __name__=="__main__":
    plot_coefficents(r, i)

