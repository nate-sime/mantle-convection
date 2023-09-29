import enum
import dataclasses
import typing

import numpy as np
import scipy
import dolfinx
import ufl


class Labels(enum.IntEnum):
    """
    Mesh labels assumed provided by the mesh generator.
    """
    # Volumes
    plate = 1
    slab = 2
    wedge = 3

    # Interfaces
    plate_wedge = 4
    slab_plate = 5
    slab_wedge = 6

    # Enclosing faces
    slab_left = 7
    slab_bottom = 8
    slab_right = 9
    wedge_right = 10
    wedge_bottom = 11
    plate_top = 12
    plate_right = 13

    # Special faces
    free_slip = 14

    def __str__(self):
        # Ensure that FFCx evaluates this as str(int()) when generating code
        return str(int(self))


# @dataclasses.dataclass
class SZData:
    """
    Utility class holding subduction zone data.
    """
    # --- Physical data
    plate_thickness: float = 50.0  # km
    Ts: float = 273.0              # Surface temperature, K
    Tmantle: float = 1573.0        # Maximum temperature, K
    k_slab: float = 3.0            # Thermal conductivity in slab, W / (m K)
    rho_slab: float = 3300.0       # Density in slab, kg / m^3
    Q_slab: float = 0.0            # Rate of heat production, W / m^3
    age_slab: float = 50e6         # Age of incoming slab, yr
    cp: float = 1250.0             # Heat capacity, J / (kg K)
    u0_m_yr: float = 0.05          # Speed scale, m / yr
    q_surf: float = 65e-3          # Surface heat flux, W / m^2
    u_conv_cm_yr: float = 5.0      # Slab convergence velocity, cm / yr
    h_r: float = 1000.0            # Length scale, m

    @property
    def overriding_side_temp(self):
        return overriding_side_temp

    def Q_wedge(self, d: np.ndarray):
        """
        Depth dependent rate of heat production in the wedge.

        Args:
            d: Depth

        Returns:
            Rate of heat production `W / m^3`
        """
        return np.zeros_like(d)

    def k_wedge(self, d: np.ndarray):
        """
        Depth dependent thermal conductivity in the wedge.

        Args:
            d: Depth

        Returns:
            Thermal conductivity `W / (m K)`
        """
        return np.full_like(d, 3.0)

    def rho_wedge(self, d: np.ndarray):
        """
        Depth dependent density in the wedge.

        Notes:
            The heat equation requires a piecewise constant density to provide
            a valid model. For spatially varying density consider conservation
            of enthalpy.

        Args:
            d: Depth

        Returns:
            Density `kg / m^3`
        """
        return np.full_like(d, 3300.0)

    def k_prime(self, k):
        """
        Scaled thermal conductivity
        """
        return k / (self.h_r * self.u_r)

    @property
    def u_r(self):
        """
        Speed scale
        """
        return self.u0_m_yr / self.t_yr_to_s(1)

    @property
    def u_conv(self):
        """
        Scaled convergence speed
        """
        return self.u_conv_cm_yr * 1e-2 / self.u0_m_yr

    @staticmethod
    def t_yr_to_s(t: float) -> float:
        """
        Conversion from years to seconds
        """
        return t * 365.0 * 24.0 * 60.0 * 60.0

    @property
    def t_r(self):
        """
        Time scale
        """
        return self.h_r / self.u_r

    def Q_prime(self, Q):
        """
        Scaled rate of heat production
        """
        return Q * (self.h_r / self.u_r)

    def t_yr_to_ndim(self, t):
        """
        Convert time from units of years to the scale of the model

        Args:
            t: time in years

        Returns:
        Scaled time
        """
        return self.t_yr_to_s(t) / self.t_r


class SZDataWilson2023(SZData):
    # --- Physical data
    plate_thickness: float = 40.0  # km
    Ts: float = 273.0              # Surface temperature, K
    Tmantle: float = 1623.0      # Maximum temperature, K
    k_slab: float = 3.1            # Thermal conductivity in slab, W / (m K)
    rho_slab: float = 3300.0       # Density in slab, kg / m^3
    Q_slab: float = 0.0            # Rate of heat production, W / m^3
    age_slab: float = 100e6        # Age of incoming slab, yr
    cp: float = 1250.0             # Heat capacity, J / (kg K)
    u0_m_yr: float = 0.05          # Speed scale, m / yr
    q_surf: float = 65e-3          # Surface heat flux, W / m^2
    u_conv_cm_yr: float = 10.0     # Slab convergence velocity, cm / yr
    h_r: float = 1000.0            # Length scale, m

    def Q_wedge(self, d: np.ndarray):
        return np.piecewise(d, [d <= 40.0, d <= 15.0], [0.27e-6, 1.3e-6, 0.0])

    def k_wedge(self, d: np.ndarray):
        return np.piecewise(d, [d <= 40.0, d <= 15.0], [2.5, 2.5, 3.1])

    def rho_wedge(self, d: np.ndarray):
        return np.piecewise(d, [d <= 40.0, d <= 15.0], [2750.0, 2750.0, 3300.0])

    @property
    def overriding_side_temp(self):
        return overriding_side_temp_heated_odeint


def create_viscosity_isoviscous() -> callable:
    def eta(u, T):
        return 1
    return eta


def create_viscosity_diffusion_creep(mesh: dolfinx.mesh.Mesh) -> callable:
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


def create_viscosity_dislocation_creep(
        mesh: dolfinx.mesh.Mesh, sz_data: SZData) -> callable:
    R = dolfinx.fem.Constant(mesh, 8.3145)
    eta_max = dolfinx.fem.Constant(mesh, 1e26)
    eta_scale = dolfinx.fem.Constant(mesh, 1e21)
    Adisl = dolfinx.fem.Constant(mesh, 28968.6)
    Edisl = dolfinx.fem.Constant(mesh, 540e3)
    n_val = dolfinx.fem.Constant(mesh, 3.5)
    n_exp = (1.0 - n_val) / n_val

    def eta(u, T):
        edot = ufl.sym(ufl.grad(u))
        eII = sz_data.u_r / sz_data.h_r * ufl.sqrt(
            0.5 * ufl.inner(edot, edot))
        eta_disl = Adisl * ufl.exp(Edisl / (n_val * R * T)) * eII ** n_exp
        eta_eff = (eta_max * eta_disl) / (eta_max + eta_disl)
        return eta_eff / eta_scale
    return eta


def gkb_wedge_flow(x: typing.Sequence[float], x0: float) -> np.ndarray:
    """
    Isoviscous wedge flow analytical solution from 'An Introduction to Fluid
    Dynamics', G.K.Batchelor. Intended for 45 degree straight downward dipping
    slab.

    Args:
        x: Spatial coordinate (x, y, 0.0)
        x0: Spatial depth of corner point

    """
    from numpy import cos, sin, arctan
    depth = -x0 - x[1]
    xdist = x[0] - x0
    xdist[np.isclose(xdist, 0.0)] = np.finfo(np.float64).eps
    values = np.zeros((2, x.shape[1]), dtype=np.float64)
    alpha = arctan(1.0)
    theta = arctan(depth / xdist)
    vtheta = (
        -((alpha - theta) * sin(theta) * sin(alpha)
          - (alpha * theta * sin(alpha - theta)))
        / (alpha ** 2 - (sin(alpha)) ** 2)
    )
    vr = (
        (((alpha - theta) * cos(theta) * sin(alpha))
         - (sin(alpha) * sin(theta))
         - (alpha * sin(alpha - theta))
         + (alpha * theta * cos(alpha - theta)))
        / (alpha ** 2 - (sin(alpha)) ** 2)
    )
    values[0, :] = - (vtheta * sin(theta) - vr * cos(theta))
    values[1, :] = - (vtheta * cos(theta) + vr * sin(theta))
    values[0, np.isclose(depth, 0.0)] = 0.0
    values[1, np.isclose(depth, 0.0)] = 0.0
    return values


def slab_inlet_temp(
        x: np.ndarray, sz_data: SZData, depth: callable,
        age_lithosphere_yr: float) -> np.ndarray:
    """
    Inlet temperature on the slab as a function of depth

    Args:
        x: Spatial coordinate
        sz_data: Slab model data
        depth: Callable function returning depth as a function of x
        age_lithosphere_yr: Age of lithosphere in units of years
    """
    vals = np.zeros((1, x.shape[1]))
    Ts = sz_data.Ts
    Tmantle = sz_data.Tmantle
    z_scale = sz_data.h_r
    kappa = sz_data.k_slab / (sz_data.rho_slab * sz_data.cp)
    vals[0] = Ts + (Tmantle - Ts) * scipy.special.erf(
        depth(x) * z_scale / (2.0 * np.sqrt(
            kappa * sz_data.t_yr_to_s(age_lithosphere_yr))))
    return vals


def overriding_side_temp(
        x: np.ndarray, sz_data: SZData, depth: callable) -> np.ndarray:
    """
    Inlet temperature on the wedge as a function of depth

    Args:
        x: Spatial coordinate
        sz_data: Slab model data
        depth: Callable function returning depth as a function of x
    """
    vals = np.zeros((1, x.shape[1]))
    Ts = sz_data.Ts
    T0 = sz_data.Tmantle
    Zplate = sz_data.plate_thickness
    vals[0] = np.minimum(Ts - (T0 - Ts) / Zplate * (-depth(x)), T0)
    return vals


def overriding_side_temp_heated_odeint(
        x: np.ndarray, sz_data: SZData, depth_f: callable,
        rtol: float = 1e2 * np.finfo(np.float64).eps,
        atol: float = 1e2 * np.finfo(np.float64).eps) -> np.ndarray:
    """
    Using SciPy's `odeint`, compute the approximate solution of

    .. math::

        - \frac{\mathrm{d}}{\mathrm{d} z}\left( k \frac{\mathrm{d}T}{\mathrm{d}z} \right) = Q

    where :math:`z` is depth, :math:`T(z=0) = T_s` and
    :math:`k \mathrm{d} T / \mathrm{d} z|_{z=0} = q`.

    Args:
        x: Position
        sz_data: Subduction zone model data
        depth_f: Depth function taking argument of position `depth_f(x)`
        rtol: Relative tolerance of the numerical ODE integration
        atol: Absolute tolerance of the numerical ODE integration

    Returns:
        Ocean-continent overriding side temperature
    """
    # Warn and ignore invalid depths
    depth = depth_f(x) * sz_data.h_r
    if np.any(depth < 0.0):
        print(f"WARNING: (min, max) depth: "
              f"({depth.min():.3e}, {depth.max():.3e})")

    depth[depth < 0.0] = 0.0

    # Second order ODE split into two first order ODEs
    def f(y, d):
        v, T = y
        Q = sz_data.Q_wedge(d / sz_data.h_r)
        k = sz_data.k_wedge(d / sz_data.h_r)
        dydz = [-Q, v / k]
        return dydz

    # Initial conditions at depth d=0
    y0 = [sz_data.q_surf, sz_data.Ts]

    # Setup depth coordinates for integration. We prepend a depth of zero to
    # provide appropriate initial conditions on all processes.
    depth_asort = np.argsort(depth)
    depth_coords = np.concatenate(([0], depth[depth_asort]))
    sol = scipy.integrate.odeint(f, y0, depth_coords, rtol=rtol, atol=atol)

    # Vector to hold temperature data
    T = np.zeros_like(x[0])

    # Transfer the solution but ignore the prepended 0 depth coordinate
    T[depth_asort] = np.minimum(sol[1:, 1], sz_data.Tmantle)
    return T
