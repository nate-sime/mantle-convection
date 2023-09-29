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


@dataclasses.dataclass
class SlabData:
    # --- Physical data
    plate_thickness: float = 40.0  # km
    Ts: float = 273.0              # Surface temperature, K
    Twedge_in: float = 1623.0      # Maximum temperature, K
    k_slab: float = 3.1            # Thermal conductivity in slab, W / (m K)
    rho_slab: float = 3300.0       # Density in slab, kg / m^3
    Q_slab: float = 0.0            # Rate of heat production, W / m^3
    cp: float = 1250.0             # Heat capacity, J / (kg K)
    u0_m_yr: float = 0.05          # Speed scale, m / yr
    q_surf: float = 65e-3          # Surface heat flux, W / m^2
    u_conv_cm_yr: float = 10.0     # Slab convergence velocity, cm / yr
    h_r: float = 1000.0            # Length scale, m

    # depth, Q, k, rho
    params_wedge: tuple = ((15e3, 1.3e-6, 2.5, 2750.0),
                           (40e3, 0.27e-6, 2.5, 2750.0),
                           (None, 0.0, 3.1, 3300.0))

    def Q_wedge(self, x, depth):
        d = depth(x)
        return np.piecewise(d, [d <= 40.0, d <= 15.0], [0.27e-6, 1.3e-6, 0.0])

    def k_wedge(self, x, depth):
        d = depth(x)
        return np.piecewise(d, [d <= 40.0, d <= 15.0], [2.5, 2.5, 3.1])

    def rho_wedge(self, x, depth):
        d = depth(x)
        return np.piecewise(d, [d <= 40.0, d <= 15.0], [2750.0, 2750.0, 3300.0])

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
        mesh: dolfinx.mesh.Mesh, slab_data: SlabData) -> callable:
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


def gkb_wedge_flow(x: typing.Sequence[float], x0: typing.Sequence[float])\
        -> np.ndarray:
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
        x: np.ndarray, slab_data: SlabData, depth: callable,
        age_lithosphere_yr: float) -> np.ndarray:
    """
    Inlet temperature on the slab as a function of depth

    Args:
        x: Spatial coordinate
        slab_data: Slab model data
        depth: Callable function returning depth as a function of x
        age_lithosphere_yr: Age of lithosphere in units of years
    """
    vals = np.zeros((1, x.shape[1]))
    Ts = slab_data.Ts
    Twedge_in = slab_data.Twedge_in
    z_scale = slab_data.h_r
    kappa = slab_data.k_slab / (slab_data.rho_slab * slab_data.cp)
    vals[0] = Ts + (Twedge_in - Ts) * scipy.special.erf(
        depth(x) * z_scale / (2.0 * np.sqrt(
            kappa * slab_data.t_yr_to_s(age_lithosphere_yr))))
    return vals


def overriding_side_temp(
        x: np.ndarray, slab_data: SlabData, depth: callable) -> np.ndarray:
    """
    Inlet temperature on the wedge as a function of depth

    Args:
        x: Spatial coordinate
        slab_data: Slab model data
        depth: Callable function returning depth as a function of x
    """
    vals = np.zeros((1, x.shape[1]))
    Ts = slab_data.Ts
    T0 = slab_data.Twedge_in
    Zplate = slab_data.plate_thickness
    vals[0] = np.minimum(Ts - (T0 - Ts) / Zplate * (-depth(x)), T0)
    return vals


def overriding_side_temp_heated(
        x: np.ndarray, slab_data: SlabData, depth: callable) -> np.ndarray:

    def T_soln(x, x0, T0, q, Q, k):
        Ta = T0 + q / k * (x - x0) - Q * (x - x0) ** 2 / (2 * k)
        qa = q - Q * (x - x0)
        return Ta, qa

    x0 = -1e-10
    q0 = slab_data.q_surf
    T0 = slab_data.Ts
    depth = depth(x) * slab_data.h_r
    if np.any(depth < -1.0):
        print(f"WARNING: (min, max) depth: {depth.min(), depth.max()}")

    depth[depth < 0.0] = 0.0
    T = np.zeros_like(x[0])
    q = np.zeros_like(x[0])
    for x_c, Q, k, rho in slab_data.params_wedge:
        if x_c is None:
            cond = (depth >= x0)
        else:
            cond = (x0 <= depth) & (depth < x_c)
        T[cond], q[cond] = T_soln(depth[cond], x0, T0, q0, Q, k)
        if x_c is None:
            break
        T0, q0 = T_soln(x_c, x0, T0, q0, Q, k)
        x0 = x_c

    T = np.minimum(T, slab_data.Twedge_in)
    # def print_depth(xc):
    #     idx = np.argmin(np.abs(depth - xc))
    #     Tbc = T[idx]
    #     print(f"Tbc({int(xc/1e3)}km) = {Tbc}(K) = {Tbc-273.0}(C), depth({int(xc/1e3)}km) = {depth[idx]}")
    # quit()
    return T  # , q