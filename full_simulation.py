import numpy as np
from dedalus import public as de
import yaml
import matplotlib.pyplot as plt
from background_field import get_background_field
from plot_parameters import set_params, get_config
set_params()


config = get_config()

n = config['domain']['num_points']
# Initialize basis and domain.
# The lower bound for r will be a small non-zero value because we don't want to deal with singularities.
r_start = config['domain']['r_start']
r_end = config['domain']['r_end']
dealias = config['domain']['dealiasing_factor']

r_basis = de.Chebyshev('r', n, interval=(r_start, r_end), dealias=dealias)
domain = de.Domain([r_basis], grid_dtype=np.float64)

r = domain.grid(0)
# Set the problem.
problem = de.InitialValueProblem(domain, variables=['xi', 'xi_t', 'xi_r'])

# Set the nondimensionalizing parameter to 1.
problem.parameters['C'] = config['free']['nondimenensionalization']

# Set the background magnetic field.
Bz0 = get_background_field(r, config)
background_field = de.Field(domain, name="Bz0")
background_field['g'] = Bz0
problem.parameters['Bz0'] = background_field

# Add main equation.
problem.add_equation("dt(xi_t) + (C / (Bz0**2 -1)) * (1 / r**2) * (-xi * (5 + Bz0**2 - 2 * r * Bz0 * dr(Bz0)) + r * xi_r * (5 + Bz0**2 + 2 * r * Bz0 * dr(Bz0)) + r**2 * dr(xi_r) * (5 + Bz0**2)) = 0")

# Add equations used for reduction.
problem.add_equation("dt(xi) - xi_t = 0")
problem.add_equation("dr(xi) - xi_r = 0")

# No movement at near the center of the machine.
boundary_velocity_magnitude = config['initialization']['velocity_magnitude']
problem.add_equation("left(xi_t) = 0")
problem.add_equation("right(xi_t) = {}".format(boundary_velocity_magnitude)) 

# Build solver.
solver = problem.build_solver(config['free']['solver']) # TODO: Figure out a good solver.

# Stopping criteria.
solver.stop_sim_time = config['domain']['end_t']
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

# Initialize the perturbation of the plasma.
xi = solver.state['xi']
xi_t = solver.state['xi_t']
xi_r = solver.state['xi_r']

boundary_position = config['initialization']['boundary_position'] # Start the wave near to the boundary.
boundary_thickness = config['initialization']['boundary_thickness']
# Initialize velocity as tanh profile on grid values.
xi_t.set_scales(1)
xi_t['g'] = boundary_velocity_magnitude * (1 + np.tanh((r - boundary_position) / boundary_thickness)) / 2

alfven_speed_at_R = (6 * Bz0[-1]**2 / (1 - Bz0[-1]**2))**0.5
# Initialize position distribution.
xi.set_scales(1)
xi['g'] = (boundary_velocity_magnitude / 2) * (boundary_thickness / alfven_speed_at_R * np.log(np.cosh((r - boundary_position) / boundary_thickness)) + (r - boundary_position) / alfven_speed_at_R + boundary_thickness / alfven_speed_at_R * np.log(2))

# Initialize position derivative.
xi.differentiate('r', out=xi_r)

# Setup storage.
xi.set_scales(1)
xi_list = [np.copy(xi['g'])]
xi_t.set_scales(1)
xi_t_list = [np.copy(xi_t['g'])]
t_list = [solver.sim_time]

# How often to save data from the simulation.
save_period = 1

# Main loop
dt = config['domain']['dt']
while solver.ok:
    solver.step(dt)

    if solver.iteration % save_period == 0:
        xi.set_scales(1)
        xi_list.append(np.copy(xi['g']))
        xi_t.set_scales(1)
        xi_t_list.append(np.copy(xi_t['g']))
        t_list.append(solver.sim_time)

    if solver.iteration % 100 == 0:
        print("Completed iteration {}.".format(solver.iteration))

# Change the various lists into arrays for easy plotting.
xi_array = np.array(xi_list)
xi_t_array = np.array(xi_t_list)
t_array = np.array(t_list)

# Calculate the contour of a particle following the alfven speed.
from particle_streamline import get_streamline
alfven_speed = (6 * Bz0**2 / (1 - Bz0**2))**0.5
r_points, t_points = get_streamline(r, -alfven_speed, r_end, r_start)

from unit_conversion import Converter
converter = Converter(config)

# plt.pcolormesh(converter.to_meter(r), converter.to_second(t_array), converter.to_meter_per_second(xi_t_array), shading='nearest', edgecolors='none')
plt.pcolormesh(converter.to_second(t_array), converter.to_meter(r), converter.to_meter_per_second(xi_t_array.T), shading='nearest', edgecolors='none')

# Plot streamlines following alfven speed.
num_alfven_curves = 10
for t_offset in np.linspace(-max(t_points), config['domain']['end_t'], num=num_alfven_curves + 2)[1:-1]:
    # plt.plot(converter.to_meter(r_points), converter.to_second(t_points + t_offset), color='red')
    plt.plot(converter.to_second(t_points + t_offset), converter.to_meter(r_points), color='red')

cbar = plt.colorbar()
cbar.set_label(r'$v \, (m/s)$')
cbar.ax.ticklabel_format(scilimits=(0, 0))
# plt.xlabel(r'$r \, (m)$')
# plt.ylabel(r'$t \, (s)$')
# plt.ylim(0, converter.to_second(config['domain']['end_t']))
plt.ylabel(r'$r \, (m)$')
plt.xlabel(r'$t \, (s)$')
plt.xlim(0, converter.to_second(config['domain']['end_t']))

plt.title("Velocity distribution evolution")
plt.tight_layout()

# plt.savefig('./plots/simulation_results.png', format='png', dpi=300)
plt.savefig('./plots/simulation_results_flipped_axis.png', format='png', dpi=300)
plt.show()
