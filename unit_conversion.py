class Converter:
    """An underscore after a parameter means it is dedimensionalized."""
    def __init__(self, config):
        self.C = config['free']['nondimenensionalization']
        self.R = config['experiment']['R_drive']
        self.T_0 = config['experiment']['temperature']
        self.B_H = config['experiment']['B_H_at_0']
        self.m_i = 1.672623099e-27 # kg
        self.m_e = 9.1093837015e-31 # kg
        self.mu_0 = 1.25663706212e-10 # H/m

        self.t_0 = (self.C * 3 * (self.m_i + self.m_e) * self.R**2 / (self.T_0))**0.5

    def to_tesla(self, B_):
        return B_ * self.B_H

    def from_tesla(self, B):
        return B / self.B_H

    def to_meter(self, r_):
        return r_ * self.R

    def from_meter(self, r):
        return r / self.R

    def to_second(self, t_):
        return t_ * self.t_0

    def from_second(self, t):
        return t / self.t_0

    def to_meter_per_second(self, v_):
        return v_ * self.R / self.t_0

    def from_meter_per_second(self, v):
        return v * self.t_0 / self.R
