from petsc4py import PETSc
import numpy as np
import dolfinx
import ufl

import mesh_generator
import model

Labels = mesh_generator.Labels


def tangent_approximation(
        V, mt: dolfinx.mesh.MeshTags, mt_id: int, tau: ufl.core.expr.Expr,
        jit_options: dict = {}, form_compiler_options: dict = {}):
    """
    Approximate the facet normal by projecting it into the function space for a
    set of facets.

    Notes:
        Adapted from `dolfinx_mpc.utils.facet_normal_approximation`.

    Args:
        V: The function space to project into
        mt: The `dolfinx.mesh.MeshTagsMetaClass` containing facet markers
        mt_id: The id for the facets in `mt` we want to represent the normal at
    """
    comm = V.mesh.comm
    n = ufl.FacetNormal(V.mesh)
    nh = dolfinx.fem.Function(V)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    ds = ufl.ds(domain=V.mesh, subdomain_data=mt, subdomain_id=mt_id)

    if V.mesh.geometry.dim == 1:
        raise ValueError("Tangent not defined for 1D problem")
    elif V.mesh.geometry.dim == 2:
        def cross_2d(u, v):
            return u[0]*v[1] - u[1]*v[0]
        b = -cross_2d(n, tau)
        sd = ufl.as_vector((n[1]*b, -n[0]*b))
        sd /= ufl.sqrt(ufl.dot(sd, sd))

        a = ufl.inner(u, v) * ds
        L = ufl.inner(sd, v) * ds
    else:
        sd = ufl.cross(n, -ufl.cross(n, tau))
        sd /= ufl.sqrt(ufl.dot(sd, sd))

        # d_x = ufl.as_vector((1, 0, 0))
        # sd_x = ufl.cross(n, -ufl.cross(n, d_x))
        # sd_x /= ufl.sqrt(ufl.dot(sd_x, sd_x))

        # sd = ufl.conditional(
        #     ufl.gt(sd[self.model.lwd[0]], 0), sd, sd_x)
        a = ufl.inner(u, v) * ds
        L = ufl.inner(sd, v) * ds

    # Find all dofs that are not boundary dofs
    imap = V.dofmap.index_map
    all_blocks = np.arange(imap.size_local, dtype=np.int32)
    top_blocks = dolfinx.fem.locate_dofs_topological(
        V, V.mesh.topology.dim - 1, mt.find(mt_id))
    deac_blocks = all_blocks[np.isin(all_blocks, top_blocks, invert=True)]

    # Note there should be a better way to do this
    # Create sparsity pattern only for constraint + bc
    bilinear_form = dolfinx.fem.form(
        a, jit_options=jit_options, form_compiler_options=form_compiler_options)
    pattern = dolfinx.fem.create_sparsity_pattern(bilinear_form)
    pattern.insert_diagonal(deac_blocks)
    pattern.finalize()
    u_0 = dolfinx.fem.Function(V)
    u_0.vector.set(0)

    bc_deac = dolfinx.fem.dirichletbc(u_0, deac_blocks)
    A = dolfinx.cpp.la.petsc.create_matrix(comm, pattern)
    A.zeroEntries()

    # Assemble the matrix with all entries
    form_coeffs = dolfinx.cpp.fem.pack_coefficients(bilinear_form._cpp_object)
    form_consts = dolfinx.cpp.fem.pack_constants(bilinear_form._cpp_object)
    dolfinx.cpp.fem.petsc.assemble_matrix(
        A, bilinear_form._cpp_object, form_consts, form_coeffs,
        [bc_deac._cpp_object])
    if bilinear_form.function_spaces[0] is bilinear_form.function_spaces[1]:
        A.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)
        A.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)
        dolfinx.cpp.fem.petsc.insert_diagonal(
            A, bilinear_form.function_spaces[0], [bc_deac._cpp_object], 1.0)
    A.assemble()
    linear_form = dolfinx.fem.form(L, jit_options=jit_options,
                            form_compiler_options=form_compiler_options)
    b = dolfinx.fem.petsc.assemble_vector(linear_form)

    dolfinx.fem.petsc.apply_lifting(b, [bilinear_form], [[bc_deac]])
    b.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, [bc_deac])

    # Solve Linear problem
    solver = PETSc.KSP().create(V.mesh.comm)
    solver.setType("cg")
    solver.rtol = 1e-8
    solver.setOperators(A)
    solver.solve(b, nh.vector)
    nh.vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    return nh


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