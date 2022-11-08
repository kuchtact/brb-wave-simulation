import numpy as np 
from dedalus import public as de
import matplotlib.pyplot as plt
from matplotlib import cm
from plasma_parameters import b_z, b_tf, alfven_speed
from plot_parameters import get_config
from coefficients import coeff_1, coeff_2, coeff_3
from screw_pinch.velocity_distribution import get_initial_velocity_distribution, get_initial_position_distribution

config = get_config()

# Parameters
n = config['domain']['num_points']
r_start = config['domain']['r_start']  # The value of r_start will be a small number close to zero to avoid any sigularities
r_end = config['domain']['r_end']
dealiasing_factor = config['domain']['dealiasing_factor']
boundary_velocity_magnitude = config['initialization']['velocity_magnitude']

# Initializing the Bases and Distributor Objects 
r_coord = de.Coordinate('r')
dist = de.Distributor(r_coord, dtype=np.float64)
r_basis = de.Chebyshev(r_coord, size=n, bounds=(r_start, r_end), dealias=dealiasing_factor)
r_grid = r_basis.global_grid(scale=1)

# Initializing the Field Objects and tau terms
xi = dist.Field(name='xi', bases=r_basis)
xi_t = dist.Field(name='xi_t', bases=r_basis)
xi_r = dist.Field(name='xi_r', bases=r_basis)
r_field = dist.Field(name='r_field', bases=r_basis)
r_field['g'] = r_grid
tau_t = dist.Field(name='tau_t')
tau_r = dist.Field(name='tau_r')

# Setting the nondimensionalization factor to 1
c = config['free']['nondimenensionalization']

# Substitutions
a1_array = coeff_1(r_grid, config)
a1 = dist.Field(name='a1', bases=r_basis)
a1['g'] = a1_array
a2_array = coeff_2(r_grid, config)
a2 = dist.Field(name='a2', bases=r_basis)
a2['g'] = a2_array
a3_array = coeff_3(r_grid, config)
a3 = dist.Field(name='a3', bases=r_basis)
a3['g'] = a3_array
Bz0_array = b_z(r_grid, config)
Bz0 = dist.Field(name='Bz0', bases=r_basis)
Bz0['g'] = Bz0_array
B_tf_array = b_tf(r_grid, config)
B_tf = dist.Field(name='B_tf', bases=r_basis)
B_tf['g'] = B_tf_array
lift = lambda A, n: de.Lift(A, r_basis, n)
dr = lambda A: de.Differentiate(A, r_coord)
print("Made substitutions")

# Setting up the problem
problem = de.IVP([xi, xi_t, xi_r, tau_t, tau_r], namespace=locals())
problem.add_equation("dt(xi_t) - c * a1 * xi - c * a2 * xi_r - c * a3 * dr(xi_r) = 0")
problem.add_equation("dt(xi) - xi_t + lift(tau_t,-1) = 0")
problem.add_equation("dr(xi) - xi_r + lift(tau_r,-1) = 0")

# Accounting for no movement at the center of the vessel
problem.add_equation("xi_t(r=r_start) = 0")
problem.add_equation("xi_t(r=r_end) = boundary_velocity_magnitude")
print("Added problem equations")

# Building the solver
solver = problem.build_solver(de.RK222)
print("Built solver")

# Setting up the stopping criteria
solver.stop_sim_time = config['domain']['end_t']
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

# Initializing the perturbation of the plasma

r = dist.local_grid(r_basis)

boundary_position = config['initialization']['boundary_position'] # Start the wave near to the boundary.
boundary_thickness = config['initialization']['boundary_thickness']

# Initializing the velocity with a hyperbolic tangent profile on the grid values
xi_t['g'] = get_initial_velocity_distribution(r, config)

alfven_speed_at_R = alfven_speed(r_grid[-1], config)

# Initializing the position distribution.
xi['g'] = get_initial_position_distribution(r, config)

# Initializing the position derivative.
xi_r = dr(xi)
print("Initialized xi")

# Change scaling of r_grid to match number of points in xi.
r_grid = r_basis.global_grid(scale=dealiasing_factor)

# Setting up the storage
xi_list = [np.copy(xi['g'])]
xi_t_list = [np.copy(xi_t['g'])]
t_list = [solver.sim_time]

# Save period for the simulation
save_period = 1

# main solver
dt = config['domain']['dt']
print("Starting solver")
while solver.proceed:
    solver.step(dt)

    if solver.iteration % save_period == 0:
        xi_list.append(np.copy(xi['g']))
            
        xi_t_list.append(np.copy(xi_t['g']))
            
        t_list.append(solver.sim_time)
            
    
    if solver.iteration % 100 == 0:
        print("Completed iteration {}.".format(solver.iteration))
print("Done solving")

# Change the various lists into arrays for easy plotting.
xi_array = np.array(xi_list[1:])
xi_t_array = np.vstack(xi_t_list[1:])
t_array = np.array(t_list[1:])

# Calculate the contour of a particle following the alfven speed.
from particle_streamline import get_streamline

# Update the Bz0_array to have the correct number of grid points.
Bz0_array = b_z(r_grid, config)
B_tf_array = b_tf(r_grid, config)
alfven_speed_array = alfven_speed(r_grid, config)
r_points, t_points = get_streamline(r_grid, -alfven_speed_array, r_end, r_start)
print("Got streamlines")

from unit_conversion import Converter
converter = Converter(config)

t_grid_si = converter.to_second(t_array)
r_grid_si = converter.to_meter(r_grid)
xi_t_grid_si = converter.to_meter_per_second(xi_t_array)

fig, (color_ax, velocity_ax) = plt.subplots(2)

mesh_plot = color_ax.pcolormesh(t_grid_si, r_grid_si, xi_t_grid_si.T, shading='nearest', edgecolors='none')
def time_to_index(t):
    return np.nonzero(t_grid_si > t)[0][0]

end_index = time_to_index(4 * 10**-6)
num_curves = 40

if end_index < num_curves:
    end_index = num_curves

color_map = cm.get_cmap('viridis', num_curves)
for cm_index, data_index in enumerate(np.linspace(0, end_index, num=num_curves, dtype=int)):
    velocity_ax.plot(r_grid_si, xi_t_grid_si[data_index], label='t={:.3f}'.format(t_grid_si[data_index]/(10**(-6))), color=color_map(cm_index))

velocity_ax.plot(converter.to_meter(r), converter.to_meter_per_second(xi_t_list[0]), color='red')

# Plot streamlines following alfven speed.
num_alfven_curves = 10
for t_offset in np.linspace(-max(t_points), config['domain']['end_t'], num=num_alfven_curves + 2)[1:-1]:
    # ax.plot(converter.to_meter(r_points), converter.to_second(t_points + t_offset), color='red')
    color_ax.plot(converter.to_second(t_points + t_offset), converter.to_meter(r_points), color='red')


cbar = fig.colorbar(mesh_plot, ax=color_ax)
cbar.set_label(r'$v \, (m/s)$')
cbar.ax.ticklabel_format(scilimits=(0, 0))
color_ax.set_ylabel(r'$r \, (m)$')
color_ax.set_xlabel(r'$t \, (s)$')
velocity_ax.set_xlabel(r'$r \, (m)$')
velocity_ax.set_ylabel(r'$v \, (m/s)$')
velocity_ax.set_title("Velocity Distribution at Different Time Slices")
# velocity_ax.legend()

color_ax.set_xlim(0, converter.to_second(config['domain']['end_t']))
color_ax.set_title("Velocity distribution evolution (TF Current = {})".format(config['coefficients']['current']))
plt.tight_layout()

plt.savefig('./plots/simulation_results_flipped_axis.png', format='png', dpi=300)
plt.show()