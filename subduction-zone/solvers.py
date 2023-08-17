from petsc4py import PETSc
import numpy as np
import dolfinx
import ufl

import mesh_generator
import model

Labels = mesh_generator.Labels

class Stokes:

    def __init__(self, mesh, facet_tags, p_order):
        self.mesh = mesh
        self.facet_tags = facet_tags
        self.p_order = p_order
        self.V = dolfinx.fem.VectorFunctionSpace(mesh, ("CG", p_order))
        self.Q = dolfinx.fem.FunctionSpace(mesh, ("CG", p_order - 1))

    def init(self, uh, Th, eta, slab_velocity):
        facet_tags = self.facet_tags
        mesh = self.mesh
        V, Q = self.V, self.Q

        u, p = map(ufl.TrialFunction, (V, Q))
        v, q = map(ufl.TestFunction, (V, Q))

        def sigma(u, u0, T):
            return 2 * eta(u0, T) * ufl.sym(ufl.grad(u))


        a_u00 = ufl.inner(sigma(u, uh, Th), ufl.sym(ufl.grad(v))) * ufl.dx
        a_u01 = - ufl.inner(p, ufl.div(v)) * ufl.dx
        a_u10 = - ufl.inner(q, ufl.div(u)) * ufl.dx
        a_u = dolfinx.fem.form(
            [[a_u00, a_u01],
             [a_u10, None]])
        a_p11 = dolfinx.fem.form(-eta(uh, Th)**-1 * ufl.inner(p, q) * ufl.dx)
        a_p = [[a_u[0][0], a_u[0][1]],
               [a_u[1][0], a_p11]]

        f_u = dolfinx.fem.Constant(mesh, [0.0] * mesh.geometry.dim)
        f_p = dolfinx.fem.Constant(mesh, 0.0)
        L_u = dolfinx.fem.form([ufl.inner(f_u, v) * ufl.dx, ufl.inner(f_p, q) * ufl.dx])


        # -- Stokes BCs
        noslip = np.zeros(mesh.geometry.dim, dtype=PETSc.ScalarType)
        noslip_facets = facet_tags.indices[
            facet_tags.values == Labels.plate_wedge]
        bc_plate = dolfinx.fem.dirichletbc(
            noslip, dolfinx.fem.locate_dofs_topological(
                V, mesh.topology.dim-1, noslip_facets), V)

        facets = facet_tags.indices[facet_tags.values == Labels.slab_wedge]
        assert isinstance(slab_velocity, (np.ndarray, dolfinx.fem.Function))
        if isinstance(slab_velocity, np.ndarray):
            bc_slab = dolfinx.fem.dirichletbc(
                slab_velocity, dolfinx.fem.locate_dofs_topological(
                    V, mesh.topology.dim-1, facets), V)
        elif isinstance(slab_velocity, dolfinx.fem.Function):
            bc_slab = dolfinx.fem.dirichletbc(
                slab_velocity, dolfinx.fem.locate_dofs_topological(
                    V, mesh.topology.dim-1, facets))


        # The plate BC goes last such that all dofs on the overriding plate
        # have priority and therefore zero flow
        self.bcs_u = [bc_slab, bc_plate]

        # -- Stokes linear system and solver
        self.a_u = a_u
        self.a_p = a_p
        self.L_u = L_u
        self.A_u = dolfinx.fem.petsc.create_matrix_block(a_u)
        self.P_u = dolfinx.fem.petsc.create_matrix_block(a_p)
        self.b_u = dolfinx.fem.petsc.create_vector_block(L_u)

        ksp_u = PETSc.KSP().create(mesh.comm)
        ksp_u.setOperators(self.A_u)
        ksp_u.setType("preonly")
        pc_u = ksp_u.getPC()
        pc_u.setType("lu")
        pc_u.setFactorSolverType("mumps")

        self.ksp_u = ksp_u
        self.x_u = self.A_u.createVecLeft()

    def assemble_stokes_system(self):
        self.A_u.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix_block(
            self.A_u, self.a_u, bcs=self.bcs_u)
        self.A_u.assemble()

        self.P_u.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix_block(
            self.P_u, self.a_p, bcs=self.bcs_u)
        self.P_u.assemble()

        with self.b_u.localForm() as b_u_local:
            b_u_local.set(0.0)
        dolfinx.fem.petsc.assemble_vector_block(
            self.b_u, self.L_u, self.a_u, bcs=self.bcs_u)

    def solve_stokes_system(self, uh):
        self.ksp_u.solve(self.b_u, self.x_u)
        offset = self.V.dofmap.index_map.size_local * self.V.dofmap.index_map_bs
        uh.x.array[:offset] = self.x_u.array_r[:offset]
        uh.x.scatter_forward()


class Heat:

    def __init__(self, mesh, facet_tags, p_order):
        self.mesh = mesh
        self.facet_tags = facet_tags
        self.p_order = p_order
        self.S = dolfinx.fem.FunctionSpace(mesh, ("CG", p_order))

    def init(self, uh, slab_data):
        facet_tags = self.facet_tags
        mesh = self.mesh
        S = self.S

        T = ufl.TrialFunction(S)
        s = ufl.TestFunction(S)

        # -- Heat system
        a_T = dolfinx.fem.form(
            ufl.inner(slab_data.k_prime * ufl.grad(T), ufl.grad(s)) * ufl.dx
            + ufl.inner(ufl.dot(slab_data.rho * slab_data.cp * uh, ufl.grad(T)), s) * ufl.dx
        )
        Q_prime_constant = dolfinx.fem.Constant(mesh, slab_data.Q_prime)
        L_T = dolfinx.fem.form(ufl.inner(Q_prime_constant, s) * ufl.dx)

        # -- Heat BCs
        def depth(x):
            return -x[1]

        inlet_facets = facet_tags.indices[facet_tags.values == Labels.slab_left]
        inlet_temp = dolfinx.fem.Function(S)
        inlet_temp.interpolate(lambda x: model.slab_inlet_temp(x, slab_data, depth))
        inlet_temp.x.scatter_forward()
        bc_inlet = dolfinx.fem.dirichletbc(
            inlet_temp, dolfinx.fem.locate_dofs_topological(
                S, mesh.topology.dim-1, inlet_facets))

        overriding_facets = facet_tags.indices[
            (facet_tags.values == Labels.plate_top) |
            (facet_tags.values == Labels.plate_right) |
            (facet_tags.values == Labels.wedge_right)]
        overring_temp = dolfinx.fem.Function(S)
        overring_temp.interpolate(
            lambda x: model.overriding_side_temp(x, slab_data, depth))
        overring_temp.x.scatter_forward()
        bc_overriding = dolfinx.fem.dirichletbc(
            overring_temp, dolfinx.fem.locate_dofs_topological(
                S, mesh.topology.dim-1, overriding_facets))

        self.bcs_T = [bc_inlet, bc_overriding]

        # -- Heat linear system and solver
        self.a_T = a_T
        self.L_T = L_T
        self.A_T = dolfinx.fem.petsc.create_matrix(self.a_T)
        self.b_T = dolfinx.fem.petsc.create_vector(self.L_T)

        ksp_T = PETSc.KSP().create(mesh.comm)
        ksp_T.setOperators(self.A_T)
        ksp_T.setType("preonly")
        pc_T = ksp_T.getPC()
        pc_T.setType("lu")
        pc_T.setFactorSolverType("mumps")

        self.ksp_T = ksp_T

    def assemble_temperature_system(self):
        self.A_T.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(self.A_T, self.a_T, bcs=self.bcs_T)
        self.A_T.assemble()

        with self.b_T.localForm() as b_T_local:
            b_T_local.set(0.0)
        dolfinx.fem.petsc.assemble_vector(self.b_T, self.L_T)
        dolfinx.fem.petsc.apply_lifting(self.b_T, [self.a_T], bcs=[self.bcs_T])
        self.b_T.ghostUpdate(addv=PETSc.InsertMode.ADD,
                        mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(self.b_T, self.bcs_T)

    def solve_temperature_system(self, Th):
        self.ksp_T.solve(self.b_T, Th.vector)
        Th.x.scatter_forward()