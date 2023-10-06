import numpy as np
import ufl
import dolfinx

import tb
import tb.utils


class ContinuousLagrange(tb.AbstractFormulation):
    def __init__(self, mesh, p):
        Ve = self.ufl_element(mesh, p)
        self.V = dolfinx.fem.FunctionSpace(mesh, Ve)

    def ufl_element(self, mesh, p):
        return ufl.FiniteElement("CG", mesh.ufl_cell(), p)

    @property
    def function_space(self):
        return self.V

    def create_soln_var(self):
        return dolfinx.fem.Function(self.V)

    def formulate(self, T, s, u):
        F = (
                ufl.dot(ufl.grad(T), ufl.grad(s))
                + ufl.dot(u, ufl.grad(T)) * s
            ) * ufl.dx
        return F

    def create_bcs(self, V):
        T_bc_top = tb.utils.create_bc_geo_marker(
            V, lambda x: np.isclose(x[1], 1.0), 0.0)
        T_bc_bot = tb.utils.create_bc_geo_marker(
            V, lambda x: np.isclose(x[1], 0.0), 1.0)

        T_bc = [T_bc_top, T_bc_bot]
        return T_bc
