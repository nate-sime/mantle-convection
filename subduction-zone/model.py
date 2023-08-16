import dataclasses
import dolfinx
import ufl


@dataclasses.dataclass
class SlabData:
    # --- Physical data
    plate_thickness: float = 50.0                  # km
    Ts: float = 273.0                              # K
    Twedge_in: float = 1573.0                      # K
    k: float = 3.0                                 # W / (m K)
    k_slab: float = 3.0                            # W / (m K)
    rho: float = 3300.0                            # kg / m^3
    rho_slab: float = 3300.0                       # kg / m^3
    cp: float = 1250.0                             # J / (kg K)
    t_yr_to_s: float = 365.0 * 24.0 * 60.0 * 60.0  # yr -> s
    u0_m_yr: float = 0.05                          # m / yr
    Q0: float = 0.0                                # W / m^3
    q_surf: float = 65e-3                          # W / m^2
    u_conv_cm_yr: float = 5.0                      # cm / yr

    # depth, Q, k, rho
    params_wedge: tuple = ((15e3, 1.3e-6, 2.5, 2700.0),
                           (40e3, 0.27e-6, 2.5, 2700.0),
                           (None, 0.0, 3.0, 3300.0))

    # Does the RHS have a piecewise heating Tbc?
    heated_wedge_bc: bool = False

    # --- Non dimensionalisation data
    h_r: float = 1000.0              # m
    t_max_yr: float = 200e6        # yr
    t_slab_max_yr: float = 100e6   # yr
    n_slab_steps: float = 20

    @property
    def k_prime(self):
        return self.k / (self.h_r * self.u_r)

    @property
    def kappa(self):
        return self.k / (self.rho * self.cp)

    @property
    def kappa_slab(self):
        return self.k_slab / (self.rho_slab * self.cp)

    @property
    def u_r(self):
        return self.u0_m_yr / self.t_yr_to_s

    @property
    def u_conv(self):
        return self.u_conv_cm_yr * 1e-2 / self.u0_m_yr

    @property
    def t50(self):
        # 50Myr to seconds
        return 50.e6 * self.t_yr_to_s

    @property
    def t_r(self):
        return self.h_r / self.u_r

    @property
    def kappa_prime(self):
        return self.kappa / (self.h_r * self.u_r)

    @property
    def Q_prime(self):
        return self.Q0 * (self.h_r / self.u_r)

    @property
    def t_max(self):
        return self.t_max_yr * self.t_yr_to_s / self.t_r

    @property
    def t_slab_max(self):
        return self.t_slab_max_yr * self.t_yr_to_s / self.t_r

    def t_yr_to_ndim(self, t):
        return t * self.t_yr_to_s / self.t_r


def create_viscosity_1():
    def eta(u, T):
        return 1
    return eta


def create_viscosity_2a(mesh):
    R = dolfinx.fem.Constant(mesh, 8.3145)
    Adiff = dolfinx.fem.Constant(mesh, 1.32043e9)
    Ediff = dolfinx.fem.Constant(mesh, 335e3)
    eta_max = dolfinx.fem.Constant(mesh, 1e26)
    eta_scale = dolfinx.fem.Constant(mesh, 1e21)
    def eta(u, T):
        eta_diff = Adiff * ufl.exp(Ediff / (R * T))
        eta_eff = (eta_max * eta_diff) / (eta_max + eta_diff)
        return eta_eff / eta_scale
    return eta


def create_viscosity_2b(mesh, slab_data):
    R = dolfinx.fem.Constant(mesh, 8.3145)
    eta_max = dolfinx.fem.Constant(mesh, 1e26)
    eta_scale = dolfinx.fem.Constant(mesh, 1e21)
    Adisl = dolfinx.fem.Constant(mesh, 28968.6)
    Edisl = dolfinx.fem.Constant(mesh, 540e3)
    n_val = dolfinx.fem.Constant(mesh, 3.5)
    n_exp = (1.0 - n_val) / n_val
    def eta(u, T):
        edot = ufl.sym(ufl.grad(u))
        eII = slab_data.u_r / slab_data.h_r * ufl.sqrt(
            0.5 * ufl.inner(edot, edot))
        eta_disl = Adisl * ufl.exp(Edisl / (n_val * R * T)) * eII ** n_exp
        eta_eff = (eta_max * eta_disl) / (eta_max + eta_disl)
        return eta_eff / eta_scale
    return eta