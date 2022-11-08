import numpy as np
from plasma_parameters import alfven_speed
from scipy.integrate import solve_ivp, trapezoid
import matplotlib.pyplot as plt


def get_initial_velocity_distribution(r_array, config):
    """
    Get the initial velocity distribution in simulation units.
    """
    r_min = config['domain']['r_start']
    r_max = config['domain']['r_end']
    r_0 = config['initialization']['boundary_position']
    delta = config['initialization']['boundary_thickness']
    magnitude = config['initialization']['velocity_magnitude']

    return magnitude * (np.tanh((r_array - r_0) / delta) - np.tanh((r_min - r_0) / delta)) / (np.tanh((r_max - r_0) / delta) - np.tanh((r_min - r_0) / delta))

def get_initial_position_distribution(r_array, config):
    """
    Get the initial position perturbation distribution in simulation units.

    Notes
    -----
    We assume that the wave velocity distribution is traveling inwards unperturbed until the simulation starts.
    We integrate over time at each `r` point starting when the wave center is far away until the wave center 
    reaches `r_0`. We'll assume the wave moves at the local alfven speed of the wave center.
    """
    # Stop when the wave gets to `10 * wave width` away from the domain edges as the wave is a constant for all 
    # `r` we care about when that far away.
    r_max = config['domain']['r_end']
    r_0 = config['initialization']['boundary_position']
    delta = config['initialization']['boundary_thickness']
    stop_event = lambda t, r: 20 * delta + r_max - r
    stop_event.terminal = True
    t_span = (0, 1)
    t_eval = np.arange(*t_span, step=10**-4)
    result = solve_ivp(lambda t, r: alfven_speed(r), t_span, np.array([r_0]), vectorized=True, dense_output=True, events=stop_event, t_eval=t_eval)
    t_points = result.t
    r_c_points = result.y[0]
    
    # Now we have the `t` and `r_c` points of the wave center. We now have to integrate the velocity distribution at each `r` in `r_array`.
    all_r_points = r_array[:, None] - r_c_points[None, :] + r_0
    all_velocity_points = get_initial_velocity_distribution(all_r_points, config)
    position_points = trapezoid(all_velocity_points, t_points, axis=-1)

    # Add the minimum if it's less than zero as the wave shouldn't move the plasma inward.
    position_points -= np.min(position_points) if np.min(position_points) < 0 else 0

    return position_points


if __name__ == "__main__":
    from plot_parameters import get_config
    config = get_config()
    r_vals = np.linspace(0, 1, num=1000)

    if True:
        # Plot initial position distribution.
        position_points = get_initial_position_distribution(r_vals, config)
        plt.plot(r_vals, position_points)
        plt.title("Initial position distribution")
        plt.xlabel(r"$r$")
        plt.ylabel(r"$\xi$")
        plt.show(block=False)
    if True:
        # Plot initial velocity distribution.
        plt.figure()
        plt.plot(r_vals, get_initial_velocity_distribution(r_vals, config))
        line_params = {'color': 'black', 'alpha': 0.2}
        plt.axvline(config['domain']['r_start'], **line_params)
        plt.axhline(0, **line_params)
        plt.axhline(config['initialization']['velocity_magnitude'], **line_params)
        plt.title("Initial velocity distribution")
        plt.xlabel(r"$r$")
        plt.ylabel(r"$\partial_t \xi$")
        plt.show(block=False)

    plt.show()
