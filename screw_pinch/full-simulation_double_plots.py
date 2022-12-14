import numpy as np 
from dedalus import public as de
import matplotlib.pyplot as plt
from matplotlib import cm
from plasma_parameters import b_z, b_tf, alfven_speed, fast_magnetosonic_speed, sound_speed
from plot_parameters import get_config
from coefficients import coeff_1, coeff_2, coeff_3
from screw_pinch.velocity_distribution import get_initial_velocity_distribution, get_initial_position_distribution
import scipy

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
t = dist.Field()
# boundary_func = lambda t: boundary_velocity_magnitude * np.sin(2 * np.pi * 5 * t) + 0.4 * boundary_velocity_magnitude * np.sin(2 * np.pi * 40 * t)
boundary_func = lambda t: boundary_velocity_magnitude * np.sin(2 * np.pi * 5 * t)
print("Made substitutions")

# Setting up the problem
problem = de.IVP([xi, xi_t, xi_r, tau_t, tau_r], namespace=locals(), time=t)
problem.add_equation("dt(xi_t) - c * a1 * xi - c * a2 * xi_r - c * a3 * dr(xi_r) = 0")
problem.add_equation("dt(xi) - xi_t + lift(tau_t,-1) = 0")
problem.add_equation("dr(xi) - xi_r + lift(tau_r,-1) = 0")

# Accounting for no movement at the center of the vessel
problem.add_equation("xi_t(r=r_start) = 0")
time_dependent_boundary = True
if time_dependent_boundary:
    problem.add_equation("xi_t(r=r_end) = boundary_func(t)")
else:
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

if time_dependent_boundary:
    xi_t['g'] = 0
    xi['g'] = 0
else:
    # Initializing the velocity with a hyperbolic tangent profile on the grid values
    xi_t['g'] = get_initial_velocity_distribution(r, config)
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
# Store the r derivative of xi_t as this is used for plotting dbz_dt.
xi_t_r_list = [np.copy(dr(xi_t).evaluate()['g'])]
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
        xi_t_r_list.append(np.copy(dr(xi_t).evaluate()['g']))
        t_list.append(solver.sim_time)
    
    if solver.iteration % 100 == 0:
        print("Completed iteration {}.".format(solver.iteration))
print("Done solving")

# Change the various lists into arrays for easy plotting.
xi_array = np.array(xi_list[1:])
xi_t_array = np.vstack(xi_t_list[1:])
xi_t_r_array = np.vstack(xi_t_r_list[1:])
t_array = np.array(t_list[1:])

# Calculate the contour of a particle following the alfven speed.
from particle_streamline import get_streamline

# Update the Bz0_array to have the correct number of grid points.
Bz0_array = b_z(r_grid, config)
B_tf_array = b_tf(r_grid, config)

alfven_speed_array = alfven_speed(r_grid, config)
alfven_r_points, alfven_t_points = get_streamline(r_grid, -alfven_speed_array, r_end, r_start)

magnetosonic_speed_array = fast_magnetosonic_speed(r_grid, config)
# magnetosonic_speed_array = sound_speed(r_grid, config)
magnetosonic_r_points, magnetosonic_t_points = get_streamline(r_grid, -magnetosonic_speed_array, r_end, r_start)
print("Got streamlines")

from unit_conversion import Converter
converter = Converter(config)

t_grid_si = converter.to_second(t_array)
r_grid_si = converter.to_meter(r_grid)
xi_t_grid_si = converter.to_meter_per_second(xi_t_array)
# Really we convert to meter per second then take a derivative in r but that's equivalent to a transformation of 1 / time.
xi_t_r_grid_si = converter.from_second(xi_t_r_array)

# PLOTTING
plot_velocity_evolution = True
plot_bdot_evolution = True
plot_discrete_velocities = False
plots_for_paper = False

if plot_velocity_evolution:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Helvetica",
    })

    fig = plt.figure()
    fig.set_size_inches(4, 3.4)

    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.08], wspace=0.06)
    (velocity_ax, vel_cbar_ax) = gs.subplots()

    time_factor = 10**6
    velocity_array = xi_t_grid_si.T * 10**-3
    velocity_plot = velocity_ax.pcolormesh(time_factor * t_grid_si, r_grid_si, velocity_array, shading='nearest', edgecolors='none', rasterized=True)
    def time_to_index(t):
        return np.nonzero(t_grid_si > t)[0][0]

    # Plot streamlines following magnetosonic and alfven speed.
    num_curves = 5
    for i, t_offset in enumerate(np.linspace(0, config['domain']['end_t'], num=num_curves + 2)[:-1]):
        # ax.plot(converter.to_meter(r_points), converter.to_second(t_points + t_offset), color='red')
        if i == 0:
            v_A_line, = velocity_ax.plot(time_factor * converter.to_second(alfven_t_points + t_offset), converter.to_meter(alfven_r_points), color='red')
            v_ms_line, = velocity_ax.plot(time_factor * converter.to_second(magnetosonic_t_points + t_offset), converter.to_meter(magnetosonic_r_points), color='orange')
        else:
            velocity_ax.plot(time_factor * converter.to_second(alfven_t_points + t_offset), converter.to_meter(alfven_r_points), color='red')
            velocity_ax.plot(time_factor * converter.to_second(magnetosonic_t_points + t_offset), converter.to_meter(magnetosonic_r_points), color='orange')

    cbar = fig.colorbar(velocity_plot, cax=vel_cbar_ax)
    cbar.set_label(r'$\partial_t \xi$ (km/s)')
    # cbar.ax.ticklabel_format(scilimits=(0, 0))
    velocity_ax.set_ylabel(r'$r$ (m)')
    velocity_ax.set_xlabel(r'$t$ ($\mu$s)')

    velocity_ax.set_xlim(0, time_factor * converter.to_second(config['domain']['end_t']))
    velocity_ax.set_title('Plasma Velocity vs. Time')
    velocity_ax.legend([v_ms_line, v_A_line], [r'$v_{ms}$', r'$v_A$'])
    
    plt.subplots_adjust(bottom=0.13, top=.93, left=0.13, right=0.88)
    plt.savefig('./plots/velocity_evolution.pdf', format='pdf', dpi=300)

if plot_bdot_evolution:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Helvetica",
    })

    fig = plt.figure()
    fig.set_size_inches(4, 3.4)

    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.08], wspace=0.06)
    (bdot_ax, bdot_cbar_ax) = gs.subplots()

    time_factor = 10**6
    
    # Convert the velocity to dB_z / dt using the frozen in field effect and density conservation.
    b_z_si = converter.to_tesla(b_z(r_grid_si))
    dr = 10**-6
    db_z_dr_si = converter.to_tesla((b_z(r_grid_si + dr) - b_z(r_grid_si - dr)) / 2)
    dbz_dt = - 1 / r_grid_si[:, None] * (xi_t_grid_si.T * b_z_si[:, None] + r_grid_si[:, None] * xi_t_r_grid_si.T * b_z_si[:, None] + r_grid_si[:, None] * xi_t_grid_si.T * db_z_dr_si[:, None])
    dbz_dt *= 10**-3 # Change to kT / s.

    # mesh_plot = plt.pcolormesh(t_grid_si, r_grid_si, dbz_dt, shading='nearest', edgecolors='none', cmap='seismic', vmin=-2000, vmax=2000)
    bdot_plot = bdot_ax.pcolormesh(time_factor * t_grid_si, r_grid_si, dbz_dt, shading='nearest', edgecolors='none', cmap='seismic', vmin=-np.max(np.abs(dbz_dt)), vmax=np.max(np.abs(dbz_dt)), rasterized=True)

    # Plot streamlines following magnetosonic and alfven speed.
    num_curves = 5
    for i, t_offset in enumerate(np.linspace(0, config['domain']['end_t'], num=num_curves + 2)[:-1]):
        # ax.plot(converter.to_meter(r_points), converter.to_second(t_points + t_offset), color='red')
        if i == 0:
            v_A_line, = bdot_ax.plot(time_factor * converter.to_second(alfven_t_points + t_offset), converter.to_meter(alfven_r_points), color='red')
            v_ms_line, = bdot_ax.plot(time_factor * converter.to_second(magnetosonic_t_points + t_offset), converter.to_meter(magnetosonic_r_points), color='orange')
        else:
            bdot_ax.plot(time_factor * converter.to_second(alfven_t_points + t_offset), converter.to_meter(alfven_r_points), color='red')
            bdot_ax.plot(time_factor * converter.to_second(magnetosonic_t_points + t_offset), converter.to_meter(magnetosonic_r_points), color='orange')

    cbar = fig.colorbar(bdot_plot, cax=bdot_cbar_ax)
    cbar.set_label(r'$\partial_t B_z$ (kT/s)')
    cbar.ax.ticklabel_format(scilimits=(0, 0))
    bdot_ax.set_ylabel(r'$r$ (m)')
    bdot_ax.set_xlabel(r'$t$ ($\mu$s)')

    bdot_ax.set_xlim(0, time_factor * converter.to_second(config['domain']['end_t']))
    bdot_ax.set_title(r'$\partial B_z / \partial t$ vs. Time')
    bdot_ax.legend([v_ms_line, v_A_line], [r'$v_{ms}$', r'$v_A$'])

    plt.subplots_adjust(bottom=0.13, top=.93, left=0.13, right=0.88)
    plt.savefig('./plots/bdot_evolution.pdf', format='pdf', dpi=300)

if plot_discrete_velocities:
    plt.figure()
    def time_to_index(t):
        return np.nonzero(t_grid_si > t)[0][0]

    end_index = time_to_index(4 * 10**-6)
    num_curves = 40

    if end_index < num_curves:
        end_index = num_curves

    color_map = cm.get_cmap('viridis', num_curves)
    for cm_index, data_index in enumerate(np.linspace(0, end_index, num=num_curves, dtype=int)):
        plt.plot(r_grid_si, xi_t_grid_si[data_index], label='t={:.3f}'.format(t_grid_si[data_index]/(10**(-6))), color=color_map(cm_index))

    plt.plot(converter.to_meter(r), converter.to_meter_per_second(xi_t_list[0]), color='red')

    plt.xlabel(r'$r \, (m)$')
    plt.ylabel(r'$v \, (m/s)$')
    plt.title("Velocity Distribution at Different Time Slices")
    # plt.legend()

    plt.savefig('./plots/discrete_velocities.png', format='png', dpi=300)

if plots_for_paper:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Helvetica",
    })

    fig = plt.figure()
    fig.set_size_inches(3.4, 6)

    gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.08], wspace=0.06, hspace=0.35)
    (velocity_ax, vel_cbar_ax), (bdot_ax, bdot_cbar_ax) = gs.subplots()

    time_factor = 10**6
    velocity_array = xi_t_grid_si.T * 10**-3
    velocity_plot = velocity_ax.pcolormesh(time_factor * t_grid_si, r_grid_si, velocity_array, shading='nearest', edgecolors='none', rasterized=True)
    def time_to_index(t):
        return np.nonzero(t_grid_si > t)[0][0]

    # Plot streamlines following magnetosonic and alfven speed.
    num_curves = 5
    for t_offset in np.linspace(0, config['domain']['end_t'], num=num_curves + 2)[1:-1]:
        # ax.plot(converter.to_meter(r_points), converter.to_second(t_points + t_offset), color='red')
        velocity_ax.plot(time_factor * converter.to_second(alfven_t_points + t_offset), converter.to_meter(alfven_r_points), color='red')
        velocity_ax.plot(time_factor * converter.to_second(magnetosonic_t_points + t_offset), converter.to_meter(magnetosonic_r_points), color='orange')

    cbar = fig.colorbar(velocity_plot, cax=vel_cbar_ax)
    cbar.set_label(r'$\partial_t \xi$ (km/s)')
    # cbar.ax.ticklabel_format(scilimits=(0, 0))
    velocity_ax.set_ylabel(r'$r$ (m)')
    velocity_ax.set_xlabel(r'$t$ ($\mu$s)')

    velocity_ax.set_xlim(0, time_factor * converter.to_second(config['domain']['end_t']))
    velocity_ax.set_title('Plasma Velocity vs. Time')
    
    # Convert the velocity to dB_z / dt using the frozen in field effect and density conservation.
    b_z_si = converter.to_tesla(b_z(r_grid_si))
    dr = 10**-6
    db_z_dr_si = converter.to_tesla((b_z(r_grid_si + dr) - b_z(r_grid_si - dr)) / 2)
    dbz_dt = - 1 / r_grid_si[:, None] * (xi_t_grid_si.T * b_z_si[:, None] + r_grid_si[:, None] * xi_t_r_grid_si.T * b_z_si[:, None] + r_grid_si[:, None] * xi_t_grid_si.T * db_z_dr_si[:, None])
    dbz_dt *= 10**-3 # Change to kT / s.

    # mesh_plot = plt.pcolormesh(t_grid_si, r_grid_si, dbz_dt, shading='nearest', edgecolors='none', cmap='seismic', vmin=-2000, vmax=2000)
    bdot_plot = bdot_ax.pcolormesh(time_factor * t_grid_si, r_grid_si, dbz_dt, shading='nearest', edgecolors='none', cmap='seismic', vmin=-np.max(np.abs(dbz_dt)), vmax=np.max(np.abs(dbz_dt)), rasterized=True)

    # Plot streamlines following magnetosonic and alfven speed.
    num_curves = 5
    for t_offset in np.linspace(0, config['domain']['end_t'], num=num_curves + 2)[1:-1]:
        # ax.plot(converter.to_meter(r_points), converter.to_second(t_points + t_offset), color='red')
        bdot_ax.plot(time_factor * converter.to_second(alfven_t_points + t_offset), converter.to_meter(alfven_r_points), color='red')
        bdot_ax.plot(time_factor * converter.to_second(magnetosonic_t_points + t_offset), converter.to_meter(magnetosonic_r_points), color='orange')

    cbar = fig.colorbar(bdot_plot, cax=bdot_cbar_ax)
    cbar.set_label(r'$\partial_t B_z$ (kT/s)')
    cbar.ax.ticklabel_format(scilimits=(0, 0))
    bdot_ax.set_ylabel(r'$r$ (m)')
    bdot_ax.set_xlabel(r'$t$ ($\mu$s)')

    bdot_ax.set_xlim(0, time_factor * converter.to_second(config['domain']['end_t']))
    bdot_ax.set_title(r'$\partial B_z / \partial t$ vs. Time')

    plt.subplots_adjust(bottom=0.07, top=.95, left=0.15, right=0.86)
    plt.savefig('./plots/vel_and_bdot.pdf', format='pdf', dpi=300)

plt.show()