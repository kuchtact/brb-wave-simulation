import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative as diff
from scipy.optimize import newton
import yaml
from plot_parameters import get_config

# Setting up the physical quatities that are constant
config = get_config()

b_helmholtz = config['experiment']['B_H_at_0']                        # The magnetic field generated due to the Helmholtz coil
mu_0 = 4 * np.pi * 10**(-7)
max_radius = config['domain']['r_end']                          # Radius of the brb
max_current = config['coefficients']['max_current']                   # Maximum current value for the torroidal field coil

r = np.linspace(config['domain']['r_start'], max_radius, config['coefficients']['r_points'])
# i = config['coefficients']['current']
dr = config['coefficients']['dr']                                # Differential step for differentiation done while calculating values for different fuctions
bounds = [(config['domain']['r_start'], max_radius)]

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
            return (mu_0 * i)/(2 * np.pi * r * b_helmholtz)
    else:
        return np.where(r==0, np.inf, (mu_0 * i )/(2 * np.pi * r * b_helmholtz))


def beta(r):
    """
    Get the plasma confinement factor at various r values

    params:
    r = radius

    """
    # return (((1 + np.cos((np.pi * r)/max_radius)) * 0.02) + 0.01)
    return (0.05 * np.exp((-r)/(max_radius**2)))

def b_z(r):
    """
    Get the residual magnetic field after the interaction
    between plasma and background magnetic field at various r values

    params:
    r = radius

    """
    # return ((b_helmholtz)/(np.sqrt(1 + beta(r))))
    return ((1)/(np.sqrt(1 + beta(r))))

# def p_0(r):
#     """
#     Get the density of the plasma at various r values

#     params:
#     r = radius values

#     """
#     return ((b_helmholtz**2 - b_z(r)**2)/(2 * mu_0))


# Calculation for getting the coefficient of radial displacement vector(\xi\)
def coeff_1(r, config):
    i = config['coefficients']['current']

    #term_1 = (-5/3) * b_helmholtz**2 + 2 * r * diff(b_tf, r, args=[i], dx=dr) * (2 * b_tf(r, i) + r * diff(b_tf, r, args=[i], dx=dr))
    term_1 = -5 + 12 * r * b_tf(r, i) * diff(b_tf, r, args=[i], dx=dr)

    #term_2 = (-1/3) * b_z(r) * (b_z(r) - 2 * r * diff(b_z, r, dx=dr))
    term_2 = 6 * r**2 * (diff(b_tf, r, args=[i], dx=dr))**2 - b_z(r)**2

    #term_3 = 2 * r**2 * b_tf(r, i) * diff(b_tf, r, n=2, args=[i], dx=dr)
    term_3 = 2 * r * b_z(r) * diff(b_z, r, dx=dr) + 6 * r**2 * b_tf(r, i) * diff(b_tf, r, n=2, args=[i], dx=dr)

    return ((1/(r**2 * (1 - b_z(r)**2))) * (term_1 + term_2 + term_3))

# Calculation for getting the coefficient of first order partial differential of the radial displacement vector(\xi\)
def coeff_2(r, config):
    i = config['coefficients']['current']

    #term_1 = (5/3) * b_helmholtz**2 + 4 * b_tf(r,i)**2 + 6 * r * b_tf(r,i) * diff(b_tf, r, args=[i], dx=dr)
    term_1 = 5 + 12 * (b_tf(r, i))**2 + 18 * r * b_tf(r, i) * diff(b_tf, r, args=[i], dx=dr)

    #term_2 = (1/3) * b_z(r) * (b_z(r) + 2 * r * diff(b_z, r, dx=dr))
    term_2 = b_z(r)**2 + 2 * r * b_z(r) * diff(b_z, r, dx=dr)
    
    return ((1/(r * (1 - b_z(r)**2))) * (term_1 + term_2))

# Calculation for getting the coefficient of second order partial differential of the radial displacement vector(\xi\)

def coeff_3(r, config):
    i = config['coefficients']['current']

    #term_1 = (5/3) * b_helmholtz**2 + 2 * b_tf(r,i)**2 + (b_z(r)**2)/3
    term_1  = 5 + 6 * (b_tf(r, i))**2 + b_z(r)**2

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

if __name__=="__main__":
    plot_coefficents(r, config)

