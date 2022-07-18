from coefficients import coeff_1 as c_1
from coefficients import coeff_2 as c_2
from coefficients import coeff_3 as c_3
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative as diff

# Defining the physical quantities that are constants

"""
Maximum value of the wavenumber(k) is given by the wavelength of the perturbation wave equal
to the electron gyroradius r = 2.38 * T^1/2 / B cm (NRL Plasma Formulatatory)

Minimum value of the wavenumber(k) is given by the wavelength of the perturbation wave equal
to the radius of brb

Parameters:
    T = 2 eV
    B = 0.01 Tesla

Conversion Factor:
    1 eV = 1.16 * 10^4 Kelvin

"""
max_radius = 0.92                                                               # Radius of the brb
min_lambda = 2.38 * 10**-2 * np.sqrt(2 * 1.16 * 10**4) / 0.01                   # Minimun value of wavelength
max_lambda = max_radius                                                         # Maximum value of wavelength
max_k = 2 * np.pi / min_lambda                                                  # Maximum value of wavenumber
min_k = 2 * np.pi / max_radius                                                  # Minimum value of wavelength
r = max_radius                                                                  # Value of the radius 
i  = 5000                                                                       # Value of the current
dk = 10**-6                                                                     # Differential step for derivative w. r. t. wavenumber
k = np.linspace(min_k, max_k, 10000)

def get_omega(k, r, i):
    """
    Get the value of the anugular frequency(omega) for various values of current, wavenumber, and, radius

    params:
    k = wavenumber values
    r = radius values
    i = current values

    """
    return (np.sqrt(k**2 * c_3(r,i) - c_1(r,i) - k * c_2(r,i) * 1j))

def get_group_velocity(k, r, i):
    """
    Get the value of the group velocity for various values of wavenumber
    using the lambda function for a given value of radius and current

    params:
    k = wavenumber values
    r = radius values
    i = current values
    
    """
    x = lambda k: get_omega(k, r, i)
    return(diff(x, k, dx=dk))

group_velocity = get_group_velocity(k, r, i)
print(group_velocity)

# Plotting the group velocity for different values of wavenumber
def plot_group_velocity():
    plt.figure()
    plt.plot(k, get_omega(k, r, i).real, color = 'r', label = 'Real part of Omega')
    plt.plot(k, get_omega(k, r, i).imag, color = 'b', label = 'Imaginary part of Omega')
    plt.plot(k, get_group_velocity(k, r, i).real, color = 'g', label = 'Real part of the group velocity')
    plt.plot(k, get_group_velocity(k, r, i).imag, label = 'Imaginary part of the group velocity')
    plt.xlabel('Wavenumber')
    plt.ylabel('Omega (angular frequency)')
    plt.title('Omega and Group Velocity')
    plt.legend()
    plt.show()

if __name__=="__main__":
    plot_group_velocity()