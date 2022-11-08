"""
Get various terms/values for the screw pinch in the BRB.
"""

import matplotlib.pyplot as plt
from scipy.misc import derivative as diff
from plot_parameters import get_config
from screw_pinch.plasma_parameters import b_tf, b_z, beta


# QUESTION: What are these commented out terms?
def coeff_1(r, config):
    """
    Calculation for getting the coefficient of radial displacement vector(xi). `xi`

    Returns
    -------
    SIMULATION
    """
    i = config['coefficients']['current']
    dr = config['coefficients']['dr']

    #term_1 = (-5/3) * b_helmholtz**2 + 2 * r * diff(b_tf, r, args=[i], dx=dr) * (2 * b_tf(r, i) + r * diff(b_tf, r, args=[i], dx=dr))
    term_1 = -5 + 12 * r * b_tf(r) * diff(b_tf, r, dx=dr)

    #term_2 = (-1/3) * b_z(r) * (b_z(r) - 2 * r * diff(b_z, r, dx=dr))
    term_2 = 6 * r**2 * (diff(b_tf, r, dx=dr))**2 - b_z(r)**2

    #term_3 = 2 * r**2 * b_tf(r, i) * diff(b_tf, r, n=2, args=[i], dx=dr)
    term_3 = 2 * r * b_z(r) * diff(b_z, r, dx=dr) + 6 * r**2 * b_tf(r) * diff(b_tf, r, n=2, dx=dr)

    return ((1/(r**2 * (1 - b_z(r)**2))) * (term_1 + term_2 + term_3))

def coeff_2(r, config):
    """
    Calculation for getting the coefficient of first order partial differential of the radial displacement vector(xi). `d/dr xi`

    Returns
    -------
    SIMULATION
    """
    dr = config['coefficients']['dr']

    #term_1 = (5/3) * b_helmholtz**2 + 4 * b_tf(r,i)**2 + 6 * r * b_tf(r,i) * diff(b_tf, r, args=[i], dx=dr)
    term_1 = 5 + 12 * (b_tf(r))**2 + 18 * r * b_tf(r) * diff(b_tf, r, dx=dr)

    #term_2 = (1/3) * b_z(r) * (b_z(r) + 2 * r * diff(b_z, r, dx=dr))
    term_2 = b_z(r)**2 + 2 * r * b_z(r) * diff(b_z, r, dx=dr)
    
    return ((1/(r * (1 - b_z(r)**2))) * (term_1 + term_2))


def coeff_3(r, config):
    """
    Calculation for getting the coefficient of second order partial differential of the radial displacement vector(xi). `d^2 / dr^2 xi`

    Returns
    -------
    SIMULATION
    """
    #term_1 = (5/3) * b_helmholtz**2 + 2 * b_tf(r,i)**2 + (b_z(r)**2)/3
    term_1  = 5 + 6 * (b_tf(r))**2 + b_z(r)**2

    return ((1/(1 - b_z(r)**2)) * term_1)

def plot_coefficents(r, config):
    # Plotting the coefficients over range of radius values for a constant value of current
    #i = config['coefficients']['current']

    c1_abs = lambda r: coeff_1(r, config)
    c2_abs = lambda r: coeff_2(r, config)
    

    # zero_c1 = newton(c1_abs, i * 10**-6)
    # print(zero_c1)

    # zero_c2 = newton(c2_abs, i * 10**-6)
    # print(zero_c2)
    plt.plot(r, beta(r))
    # # plt.plot(r, coeff_1(r,config), color = 'r', label = 'coefficient 1')
    # # plt.plot(r, coeff_2(r,config), color = 'g', label = 'coefficient 2')
    # # plt.plot(r, coeff_3(r,config), color = 'b', label = 'coefficient 3')
    # plt.axhline(0, color = 'black')
    # # plt.plot(zero_c1, 0, marker='o')
    # # plt.plot(zero_c2, 0, marker='o')
    # # plt.ylim(-1000000000000, 1000000000000)
    # plt.ylim(-10**4, 10**4)
    # plt.xlabel('Radius')
    # plt.ylabel('Coefficients')
    # plt.title("TF Current = {current} A".format(current=config['coefficients']['current']))

    plt.legend()

    plt.show()

