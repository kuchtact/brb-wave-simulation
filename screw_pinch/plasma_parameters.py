import numpy as np
from unit_conversion import Converter
from plot_parameters import get_config
mu_0 = 4 * np.pi * 10**(-7)
default_config = get_config()
gamma = 5 / 3


def b_tf(r, config=None, i=None):
    """
    Get the torroidal field for various r and i values in experimental units.

    Parameters
    ----------
    r : np.array or float
        `r` positions in simulation units.
    config : dict or None
        The config dictionary. If `None` then use the default config.
    i : float, default=None
        Current in experimental units passing through the TF coil. If `None` then load from config.

    Returns
    -------
    SIMULATION
    """
    if config is None:
        config = default_config

    if i is None:
        i = config['coefficients']['current']

    if isinstance(r, float):
        if r==0:
            return np.inf
        else:
            return (mu_0 * i)/(2 * np.pi * r * config['experiment']['B_H_at_0'])
    else:
        return np.where(r==0, np.inf, (mu_0 * i)/(2 * np.pi * r * config['experiment']['B_H_at_0']))

def beta(r, config=None):
    """
    Get the plasma beta.

    Parameters
    ----------
    r : np.array
        `r` positions in simulation units. `r = 1` is the position of `beta_at_R`.
    config : dict or None
        The config dictionary. If `None` then use the default config.

    Returns
    -------
    SIMULATION
    """
    if config is None:
        config = default_config

    if config['background']['profile'] == 'cos':
        beta = (1 + np.cos(np.pi * r)) * (config['background']['beta_at_0'] - config['background']['beta_at_R']) / 2 + config['background']['beta_at_R']
    elif config['background']['profile'] == 'gaussian':
        A = config['background']['beta_at_0']
        B = -1 / config['domain']['r_end']**2 * np.log(config['background']['beta_at_R'] / config['background']['beta_at_0'])
        beta = A * np.exp(-B * r**2)
    else:
        raise ValueError("The `background/profile` in the config file is not allowed. Possible profiles are 'cos' and 'gaussian'.")

    return beta

def b_z(r, config=None):
    """
    Get the residual magnetic field after the interaction
    between plasma and background magnetic field at various r values

    Parameters
    ----------
    r : np.array of float
        `r` positions in simulation units.
    config : dict or None
        The config dictionary. If `None` then use the default config.

    Returns
    -------
    SIMULATION
    """
    if config is None:
        config = default_config

    return 1 / np.sqrt(1 + beta(r, config))

def p_0(r, config=None):
    """
    Get the plasma pressure in experimental units.

    Parameters
    ----------
    r : np.array of float
        `r` position in simulation units.
    config : dict or None
        The config dictionary. If `None` then use the default config.

    Returns
    -------
    EXPERIMENTAL
    """
    if config is None:
        config = default_config

    converter = Converter(config)
    return beta(r, config) * converter.to_tesla(b_z(r, config))**2 / (2 * mu_0)

def n_0(r, config=None):
    """
    Get the plasma number density in experimental units assuming constant temperature.
    
    Parameters
    ----------
    r : np.array of float
        `r` position in simulation units.
    config : dict or None
        The config dictionary. If `None` then use the default config.

    Returns
    -------
    EXPERIMENTAL
    """
    if config is None:
        config = default_config

    return p_0(r, config) / config['experiment']['temperature']

def rho_0(r, config=None):
    """
    Get the plasma mass density in experimental units assuming constant temperature.
    
    Parameters
    ----------
    r : np.array of float
        `r` position in simulation units.
    config : dict or None
        The config dictionary. If `None` then use the default config.

    Returns
    -------
    EXPERIMENTAL
    """
    if config is None:
        config = default_config

    return n_0(r, config) * config['experiment']['ion_mass']

def alfven_speed(r, config=None):
    """
    Get the alfven speed.

    Parameters
    ----------
    r : np.array of float
        `r` position in simulation units.
    config : dict or None
        The config dictionary. If `None` then use the default config.

    Returns
    -------
    SIMULATION
    """
    if config is None:
        config = default_config

    return (6 * (b_z(r, config)**2 + b_tf(r, config)**2) / (1 - b_z(r, config)**2))**0.5

def sound_speed(r, config=None):
    """
    Get the sound speed.

    Parameters
    ----------
    r : np.array of float
        `r` position in simulation units.
    config : dict or None
        The config dictionary. If `None` then use the default config.

    Returns
    -------
    SIMULATION

    Notes
    -----
    We've used the calculation that can be found here: https://farside.ph.utexas.edu/teaching/plasma/lectures/node65.html
    Since p_0 = n_0 T_0 and rho_0 = (m_i + m_e) * n_0 ~= m_i * n_0 then v_s ~= (gamma * T_0 / m_i)**0.5
    """
    if config is None:
        config = default_config

    converter = Converter(config)

    return converter.from_meter_per_second((gamma * config['experiment']['temperature'] / config['experiment']['ion_mass'])**0.5) * np.ones_like(r)

def fast_magnetosonic_speed(r, config=None, theta=np.pi / 2):
    """
    Get the speed of the fast magnetosonic wave.

    Parameters
    ----------
    r : np.array of float
        `r` position in simulation units.
    config : dict or None
        The config dictionary. If `None` then use the default config.
    theta : float, default=np.pi / 2
        Angle in radians between wave vector and background field.

    Returns
    -------
    SIMULATION
    """
    if config is None:
        config = default_config

    v_a = alfven_speed(r, config)
    v_s = sound_speed(r, config)
    return (0.5 * (v_a**2 + v_s**2 + ((v_a**2 + v_s**2)**2 - 4 * v_a**2 * v_s**2 * np.cos(theta)**2)**0.5))**0.5
