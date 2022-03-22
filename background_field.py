from scipy.integrate import quad
import numpy as np
from plot_parameters import get_config


def get_helmholtz_field(r_array, config):
    """
    Get the helmholtz field at various r points.
    
    params:
    r_array = Dedimensionalized radius.
    config  = Loaded config file.
    """
    R_H = config['experiment']['R_helmholtz']
    R_drive = config['experiment']['R_drive']
    L_H = config['experiment']['L_helmholtz']

    # Setup the integral.
    integrand = lambda theta, r: (1 - r * np.cos(theta)) / (r**2 + 1 + (L_H / (2 * R_H))**2 - 2 * r * np.cos(theta))**(3/2)
    integral = lambda r: (R_H**2 + (L_H / 2)**2)**(3/2) / (np.pi * R_H**3) * quad(integrand, 0, np.pi, args=(r))[0]

    # Vectorize the integral so that we can do it for each point in the r array.
    vectorized_integral = np.vectorize(integral)

    helmholtz_field = vectorized_integral(r_array * R_drive / R_H)
    # TODO: Change the math to allow a non-constant helmholtz field.
    return np.ones_like(r_array)

def get_beta_correction_factor(r_array, config):
    """Get the factor to correct the helmholtz field by."""
    beta = (1 + np.cos(np.pi * r_array)) * (config['background']['beta_at_0'] - config['background']['beta_at_R']) / 2 + config['background']['beta_at_R']
    factor = 1 / (beta + 1)**0.5
    return factor

def get_background_field(r_array, config):
    helmholtz = get_helmholtz_field(r_array, config)

    background = helmholtz * get_beta_correction_factor(r_array, config)
    return background


if __name__=="__main__":
    config = get_config()
    r_values = np.linspace(0, 1.5)
    B_H = get_helmholtz_field(r_values, config)

    import matplotlib.pyplot as plt
    plt.plot(r_values, B_H)
    plt.show()
