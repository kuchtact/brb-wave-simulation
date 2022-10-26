from numbers import Integral
import numpy as np
from dedalus import public as de
import yaml
import matplotlib.pyplot as plt
from background_field import get_background_field
from plot_parameters import set_params, get_config
# set_params()


config = get_config()

n = config['domain']['num_points']
# Initialize basis and domain.
# The lower bound for r will be a small non-zero value because we don't want to deal with singularities.
r_start = config['domain']['r_start']
r_end = config['domain']['r_end']
dealias_factor = config['domain']['dealiasing_factor']

# r_basis = de.Chebyshev('r', n, interval=(r_start, r_end), dealias=dealias_factor)
# domain = de.Domain([r_basis], grid_dtype=np.float64)

# Bases
rcoord = de.Coordinate('r')
dist = de.Distributor(rcoord, dtype=np.float64)
rbasis = de.Chebyshev(rcoord, size=n, bounds=(r_start,r_end), dealias=dealias_factor)
r_grid = rbasis.global_grid(scale=1)
# r_field = dist.Field(name='r_field', bases=rbasis)
# r_field['g'] = r_grid

# Fields
xi = dist.Field(name='xi', bases=rbasis)
xi_t = dist.Field(name='xi_t', bases=rbasis)
xi_r = dist.Field(name='xi_r', bases=rbasis)
#tau_xi = dist.Field(name='tau_xi')
tau_r = dist.Field(name='tau_r')
tau_t = dist.Field(name='tau_t')
r_field = dist.Field(name='r_field', bases=rbasis)
r_field['g'] = r_grid

#r = domain.grid(0)
# Set the problem.
#problem = de.InitialValueProblem([xi, xi_t, xi_r, tau_xi, tau_t, tau_r], namespace=locals())
problem = de.IVP([xi, xi_t, xi_r, tau_t, tau_r], namespace=locals())


# Set the nondimensionalizing parameter to 1.
c = config['free']['nondimenensionalization']

# Substitutions
Bz0_array = get_background_field(r_grid, config)
Bz0 = dist.Field(name='Bz0', bases=rbasis)
Bz0['g'] = Bz0_array
#problem.parameters['Bz0'] = background_field
lift = lambda A, n: de.Lift(A, rbasis, n)
dr = lambda A: de.Differentiate(A, rcoord)


# Add main equation.
# problem.add_equation("dt(xi_t) + (1 / r_field**2) * (-xi) + lift(tau_xi,-1) = 0")
#problem.add_equation("dt(xi_t) + (c / (Bz0**2 -1)) * (1 / r_field**2) * (-xi * (5 + Bz0**2 - 2 * r_field * Bz0 * dr(Bz0)) + r_field * xi_r * (5 + Bz0**2 + 2 * r_field * Bz0 * dr(Bz0)) + r_field**2 * dr(xi_r) * (5 + Bz0**2)) + lift(tau_xi,-1) = 0")
#problem.add_equation("dt(xi_t) + dr(xi_r) + xi_r + xi = 0")
problem.add_equation("dt(xi_t) + (c / (Bz0**2 -1)) * (1 / r_field**2) * (-xi * (5 + Bz0**2 - 2 * r_field * Bz0 * dr(Bz0)) + r_field * xi_r * (5 + Bz0**2 + 2 * r_field * Bz0 * dr(Bz0)) + r_field**2 * dr(xi_r) * (5 + Bz0**2)) = 0")
# Add equations used for reduction.
#problem.add_equation("integ(xi) + tau_xi = 0")
problem.add_equation("dt(xi) - xi_t + lift(tau_t,-1) = 0")
problem.add_equation("dr(xi) - xi_r + lift(tau_r,-1) = 0")

# No movement at near the center of the machine.
boundary_velocity_magnitude = config['initialization']['velocity_magnitude']
problem.add_equation("xi_t(r=r_start) = 0")
#problem.add_equation("right(xi_t) = {}".format(boundary_velocity_magnitude)) 
problem.add_equation("xi_t(r=r_end) = boundary_velocity_magnitude")

# Build solver.
# TODO: Move the definition of the solver to the config.
solver = problem.build_solver(de.RK222) # TODO: Figure out a good solver.

# Stopping criteria.
solver.stop_sim_time = config['domain']['end_t']
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

# Initialize the perturbation of the plasma.

r = dist.local_grid(rbasis)

boundary_position = config['initialization']['boundary_position'] # Start the wave near to the boundary.
boundary_thickness = config['initialization']['boundary_thickness']
# Initialize velocity as tanh profile on grid values.
xi_t['g'] = boundary_velocity_magnitude * (1 + np.tanh((r - boundary_position) / boundary_thickness)) / 2

alfven_speed_at_R = (6 * Bz0_array[-1]**2 / (1 - Bz0_array[-1]**2))**0.5
# Initialize position distribution.
xi['g'] = (boundary_velocity_magnitude / 2) * (boundary_thickness / alfven_speed_at_R * np.log(np.cosh((r - boundary_position) / boundary_thickness)) + (r - boundary_position) / alfven_speed_at_R + boundary_thickness / alfven_speed_at_R * np.log(2))

# Initialize position derivative.
xi_r = dr(xi)


# Setup storage.

xi_list = [np.copy(xi['g'])]

xi_t_list = [np.copy(xi_t['g'])]
t_list = [solver.sim_time]
# Change scaling of r_grid to match number of points in xi.
r_grid = rbasis.global_grid(scale=dealias_factor)

# How often to save data from the simulation.
save_period = 1
i = 0
j = 0
k = 0

# Main loop
dt = config['domain']['dt']
while solver.proceed:
    solver.step(dt)

    if solver.iteration % save_period == 0:
        xi_list.append(np.copy(xi['g']))
            
        xi_t_list.append(np.copy(xi_t['g']))
            
        t_list.append(solver.sim_time)
            

    if solver.iteration % 100 == 0:
        print("Completed iteration {}.".format(solver.iteration))

# Change the various lists into arrays for easy plotting.
xi_list.pop(0)
xi_array = np.array(xi_list)
xi_t_list.pop(0)
xi_t_array = np.vstack(xi_t_list)
t_list.pop(0)
t_array = np.array(t_list)

# Calculate the contour of a particle following the alfven speed.
from particle_streamline import get_streamline
# Update the Bz0_array to have the correct number of grid points.
Bz0_array = get_background_field(r_grid, config)
alfven_speed = (6 * Bz0_array**2 / (1 - Bz0_array**2))**0.5
r_points, t_points = get_streamline(r_grid, -alfven_speed, r_end, r_start)

from unit_conversion import Converter
converter = Converter(config)

t_array_si = converter.to_second(t_array)
r_grid_si = converter.to_meter(r_grid)
xi_t_array_si = converter.to_meter_per_second(xi_t_array)
# plt.pcolormesh(converter.to_meter(r_grid), converter.to_second(t_array), converter.to_meter_per_second(xi_t_array), shading='nearest', edgecolors='none')
plt.pcolormesh (t_array_si, r_grid_si, xi_t_array_si.T, shading='nearest', edgecolors='none')

# Plot streamlines following alfven speed.
num_alfven_curves = 10
for t_offset in np.linspace(-max(t_points), config['domain']['end_t'], num=num_alfven_curves + 2)[1:-1]:
    #  plt.plot(converter.to_meter(r_points), converter.to_second(t_points + t_offset), color='red')
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
