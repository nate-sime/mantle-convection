import numpy as np
import ufl
import dolfinx

import tb
import tb.utils


class TaylorHood(tb.AbstractFormulation):
    def __init__(self, mesh, p):
        W = dolfinx.fem.FunctionSpace(
            mesh, ufl.MixedElement(self.ufl_element(mesh, p)))
        self.W = W

    def ufl_element(self, mesh, p):
        Ve_vec = ufl.VectorElement("CG", mesh.ufl_cell(), p)
        Qe = ufl.FiniteElement("CG", mesh.ufl_cell(), p - 1)
        return [Ve_vec, Qe]

    @property
    def function_space(self):
        return self.W

    def create_soln_var(self):
        return dolfinx.fem.Function(self.W)

    def velocity(self, soln_var):
        return soln_var.sub(0)

    def velocity_for_output(self, soln_var):
        return soln_var.sub(0).collapse()

    def formulate(self, mu, f, soln_var):
        u, p = ufl.split(soln_var)
        v, q = ufl.TestFunctions(soln_var.function_space)

        F = (
            ufl.inner(2 * mu * ufl.sym(ufl.grad(u)), ufl.grad(v))
            - ufl.inner(p, ufl.div(v)) - ufl.inner(q, ufl.div(u))
            - ufl.inner(f, v)
        ) * ufl.dx
        return F

    def create_bcs(self, W):
        Vvel = W.sub(0).sub(0).collapse()
        zero_u = dolfinx.fem.Function(Vvel[0])
        zero_bc_x = tb.utils.create_bc_geo_marker(
            (W.sub(0).sub(0), Vvel[0]),
            lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0), zero_u)

        Vvel = W.sub(0).sub(1).collapse()
        zero_v = dolfinx.fem.Function(Vvel[0])
        zero_bc_y = tb.utils.create_bc_geo_marker(
            (W.sub(0).sub(1), Vvel[0]),
            lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0), zero_v)

        Q = W.sub(1).collapse()
        zero_p = dolfinx.fem.Function(Q[0])
        dofs_p = dolfinx.fem.locate_dofs_topological(
            (W.sub(1), Q[0]), 0, [0])
        zero_p_bc = dolfinx.fem.dirichletbc(zero_p, dofs_p, W.sub(1))

        u_bcs = [zero_bc_x, zero_bc_y, zero_p_bc]
        return u_bcs


class C0_SIPG(tb.AbstractFormulation):

    @staticmethod
    def tensor_jump(u, n):
        return ufl.outer(u, n)("+") + ufl.outer(u, n)("-")

    @staticmethod
    def G_mult(G, tau):
        m, d = tau.ufl_shape
        return ufl.as_matrix([[ufl.inner(G[i, k, :, :], tau) for k in range(d)]
                              for i in range(m)])

    def __init__(self, mesh, p):
        self.p = p
        self.V = dolfinx.fem.FunctionSpace(mesh, self.ufl_element(mesh, p))

    def ufl_element(self, mesh, p):
        return ufl.FiniteElement("CG", mesh.ufl_cell(), p)

    @property
    def function_space(self):
        return self.V

    def create_soln_var(self):
        return dolfinx.fem.Function(self.V)

    def velocity(self, soln_var):
        return ufl.curl(soln_var)

    def velocity_for_output(self, soln_var):
        mesh = soln_var.function_space.mesh
        uh = dolfinx.fem.Function(dolfinx.fem.VectorFunctionSpace(
            mesh, ("DG", self.p - 1)))
        uh.interpolate(dolfinx.fem.Expression(
            self.velocity(soln_var),
            uh.function_space.element.interpolation_points()))
        return uh

    def formulate(self, mu, f, soln_var):
        mesh = soln_var.function_space.mesh
        h = ufl.CellVolume(mesh) / ufl.FacetArea(mesh)
        n = ufl.FacetNormal(mesh)
        penalty_constant = 20.0
        beta = dolfinx.fem.Constant(mesh, penalty_constant * self.p ** 2) / h

        G = mu * ufl.as_tensor([[
            [[2, 0],
             [0, 0]],
            [[0, 1],
             [1, 0]]],
            [[[0, 1],
              [1, 0]],
             [[0, 0],
              [0, 2]]]])

        def sigma(u):
            return 2 * mu * ufl.sym(ufl.grad(u))

        def Bh(u, v):
            domain = ufl.inner(sigma(u), ufl.sym(ufl.grad(v))) * ufl.dx
            interior = (
                - ufl.inner(self.tensor_jump(u, n), ufl.avg(sigma(v)))
                - ufl.inner(self.tensor_jump(v, n), ufl.avg(sigma(u)))
                + ufl.inner(beta("+") * self.G_mult(ufl.avg(G), self.tensor_jump(u, n)),
                            self.tensor_jump(v, n))
            ) * ufl.dS
            return domain + interior# + exterior

        def lh(v):
            domain = ufl.inner(f, v) * ufl.dx
            return domain

        u = self.velocity(soln_var)
        v = self.velocity(ufl.TestFunction(self.function_space))
        return Bh(u, v) - lh(v)

    def create_bcs(self, V):
        mesh = V.mesh
        facets = dolfinx.mesh.locate_entities_boundary(
            mesh, dim=mesh.topology.dim - 1,
            marker=lambda x: np.ones_like(x[0], dtype=np.int8))
        dofs = dolfinx.fem.locate_dofs_topological(
            V, mesh.topology.dim - 1, facets)
        zero_bc = dolfinx.fem.dirichletbc(0.0, dofs, V)
        return [zero_bc]
