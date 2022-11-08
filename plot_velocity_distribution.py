import numpy as np
from dedalus import public as de
import yaml
import matplotlib.pyplot as plt
from plot_parameters import set_params, get_config
set_params()


config = get_config()

n = config['domain']['num_points']
r_start = config['domain']['r_start']
r_end = config['domain']['r_end']
dealias = config['domain']['dealiasing_factor']

# r_basis = de.Chebyshev('r', n, interval=(r_start, r_end), dealias=dealias)
# domain = de.Domain([r_basis], grid_dtype=np.float64)

rcoord = de.Coordinate('r')
dist = de.Distributor(rcoord, dtype=np.float64)
rbasis = de.Chebyshev(rcoord, size=n, bounds=(r_start,r_end), dealias=dealias)


# r = domain.grid(0)
# xi_t = de.Field(domain, name='xi_t', scales=1)
r = dist.local_grid(rbasis)
xi_t = dist.Field(name='xi_t', bases=rbasis)

boundary_velocity_magnitude = config['initialization']['velocity_magnitude']
boundary_position = config['initialization']['boundary_position'] # Start the wave near to the boundary.
boundary_thickness = config['initialization']['boundary_thickness']
xi_t['g'] = boundary_velocity_magnitude * (1 + np.tanh((r - boundary_position) / boundary_thickness)) / 2

ax = plt.axes()
ax.set_xlabel(r"$\tilde{r}$")
ax.set_ylabel(r"$\partial_\tau \tilde{\xi}$")
ax.set_xticks(np.array(r))
ax.set_xticklabels(['${}$'.format(r_start)] + [None] * (len(r) - 2) + ['${}$'.format(r_end)])
ax.set_xlim(0, 1)
ax.set_title("Initial velocity distribution")
plt.plot(r, xi_t['g'], color='blue', alpha=0.5)
plt.scatter(r, xi_t['g'], color='red', s=5)

plt.axvline(boundary_position, color='#0b8700', lw=1, label=r'$\tilde{r}_0$')
plt.axvline(boundary_position + boundary_thickness, color='#17fc03', lw=1)
plt.axvline(boundary_position - boundary_thickness, color='#17fc03', lw=1, label=r'$\tilde{r}_0 \pm \delta$')

plt.legend()
plt.savefig('./plots/velocity_distribution.pdf', format='pdf', dpi=300)
plt.show()
