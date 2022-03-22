import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad


def get_streamline(x_array, v_array, x_start, x_end, num=100):
    """
    Get the movement of a particle placed at some point in the velocity field.
    
    params:
    x_array                 = Positions of each velocity.
    v_array                 = Velocity at each point.
    x_start                 = Starting point of particle.
    x_end                   = Ending point of particle. If it can't get there then return error.
    num                     = Number of points to keep.
    """
    # Interpolate the velocity so we can calculate the velocity at any point when doing our integral.
    v_interpolator = interp1d(x_array, v_array, bounds_error=False, fill_value=(v_array[0], v_array[-1]))
    if not valid_points(x_array, v_interpolator, x_start, x_end):
        raise ValueError("Could not find path from start to end point.")

    # Positions of particle over time.
    x_points = np.linspace(x_start, x_end, num=num)
    # Hold how long it took to get to the new position.
    t_points = np.zeros(num)

    curr_t = 0
    for i in range(1, num):
        # Integrate over 1 / v for each step in x. This gives us the time it takes for the particle to move that distance.
        dt, _ = quad(lambda x: 1 / v_interpolator(x), x_points[i - 1], x_points[i])
        # Add the time taken to the current time and record the time.
        curr_t += dt
        t_points[i] = curr_t

    return x_points, t_points

def valid_points(x_array, v_interpolator, x_start, x_end):
    """
    Determine whether two points can be connected by a streamline.
    """
    direction = np.sign(x_end - x_start)

    if v_interpolator(x_start) * direction < 0 or v_interpolator(x_end) * direction < 0:
        return False
    
    # Go through each point between the start and end points. If the direction is ever wrong then we can't get from one to the other.
    for x in x_array:
        if x < min(x_start, x_end):
            continue

        if v_interpolator(x) * direction < 0 or v_interpolator(x) == 0:
            return False
        
        if x < max(x_start, x_end):
            break
    
    return True
