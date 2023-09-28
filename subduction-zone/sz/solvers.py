import typing

from petsc4py import PETSc
import numpy as np
import dolfinx
import ufl

import sz
import sz.model

Labels = sz.model.Labels


def steepest_descent(
        V: dolfinx.fem.FunctionSpace,
        tau: ufl.core.expr.Expr,
        plate_depth: typing.Optional[float] = None,
        couple_depth: typing.Optional[float] = None,
        depth: typing.Callable[
            [np.ndarray | ufl.core.expr.Expr],
            np.ndarray | ufl.core.expr.Expr] = None,):
    """
    Formulate the vector which lies tangential to a surface and maximises
    the scalar projection with the vector `tau`. If the optional vector `d`
    is provided, modify the projected vector such that it points in the
    direction such that its scalar product with `d` is positive.

    Args:
        V: The function space into which to project
        tau: The direction of the tangent alignment to maximise
        plate_depth: Depth of the plate
        couple_depth: Depth of the coupling point
        depth: Function taking position x and returning depth

    Returns:
        UFL expression
    """
    n = ufl.FacetNormal(V.mesh)
    if V.mesh.geometry.dim == 1:
        raise ValueError("Tangent not defined for 1D problem")
    elif V.mesh.geometry.dim == 2:
        def cross_2d(u, v):
            return u[0]*v[1] - u[1]*v[0]
        b = -cross_2d(n, tau)
        sd = ufl.as_vector((n[1]*b, -n[0]*b))
    else:
        sd = ufl.cross(n, -ufl.cross(n, tau))
    sd /= ufl.sqrt(ufl.dot(sd, sd))

    if plate_depth is not None and couple_depth is not None and depth is not None:
        x = ufl.SpatialCoordinate(V.mesh)
        sd *= ufl.max_value(
            0.0, ufl.min_value(
                1.0,
                (depth(x) - plate_depth) / (couple_depth - plate_depth)))
    return sd


def facet_local_projection(
        V: dolfinx.fem.FunctionSpace,
        mt: dolfinx.mesh.MeshTags, mt_id: int | typing.Iterable[int],
        f: ufl.core.expr.Expr,
        jit_options: typing.Optional[dict] = {},
        form_compiler_options: typing.Optional[dict] = {}) \
        -> dolfinx.fem.Function:
    """
    Approximate the facet normal by projecting it into the function space for a
    set of facets. If a coupling depth is provided, the ve

    Notes:
        Adapted from `dolfinx_mpc.utils.facet_normal_approximation`.

    Args:
        V: The function space into which to project
        mt: Facet markers
        mt_id: The id for the facets in `mt` from which the topological DoFs
         in V will be populated
        jit_options: FFCx form JIT options
        form_compiler_options: FFCx form compiler options

    Returns
        Facet local projection in a `dolfinx.fem.Function` defined on `V`
    """
    comm = V.mesh.comm
    mt_id_tuple = (mt_id,) if not hasattr(mt_id, "__len__") else tuple(mt_id)
    f_h = dolfinx.fem.Function(V)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    ds = ufl.ds(domain=V.mesh, subdomain_data=mt, subdomain_id=mt_id_tuple)

    a = ufl.inner(u, v) * ds
    L = ufl.inner(f, v) * ds

    # Find all dofs that are not boundary dofs
    imap = V.dofmap.index_map
    all_blocks = np.arange(imap.size_local, dtype=np.int32)
    top_blocks = dolfinx.fem.locate_dofs_topological(
        V, V.mesh.topology.dim - 1, mt.indices[np.isin(mt.values, mt_id_tuple)])
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
    solver.solve(b, f_h.vector)
    f_h.vector.ghostUpdate(
        addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    return f_h


class Stokes:

    def __init__(self, mesh, facet_tags, p_order):
        self.mesh = mesh
        self.facet_tags = facet_tags
        self.p_order = p_order
        self.V = dolfinx.fem.FunctionSpace(
            mesh, ("CG", p_order, (mesh.geometry.dim,)))
        self.Q = dolfinx.fem.FunctionSpace(mesh, ("CG", p_order - 1))

    def init(self, uh, Th, eta, slab_velocity, use_iterative_solver):
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

        # Free slip terms
        n = ufl.FacetNormal(mesh)
        h = ufl.CellDiameter(mesh)
        ds = ufl.Measure("ds", subdomain_data=facet_tags)
        alpha = dolfinx.fem.Constant(mesh, 20.0 * V.ufl_element().degree() ** 2)
        ds_fs = ds(Labels.free_slip)
        a_u00 += - ufl.inner(sigma(u, uh, Th), ufl.outer(ufl.dot(v, n) * n, n)) * ds_fs\
                 - ufl.inner(ufl.outer(ufl.dot(u, n) * n, n), sigma(v, uh, Th)) * ds_fs\
                 + 2 * eta(uh, Th) * alpha / h * ufl.inner(ufl.outer(ufl.dot(u, n) * n, n), ufl.outer(v, n)) * ds_fs
        a_u01 += ufl.inner(p, ufl.dot(v, n)) * ds_fs
        a_u10 += ufl.inner(ufl.dot(u, n), q) * ds_fs

        a_u = dolfinx.fem.form(
            [[a_u00, a_u01],
             [a_u10, None]])
        a_p11 = dolfinx.fem.form(-eta(uh, Th)**-1 * ufl.inner(p, q) * ufl.dx)
        a_p = [[a_u[0][0], a_u[0][1]],
               [a_u[1][0], a_p11]]

        f_u = dolfinx.fem.Constant(mesh, [0.0] * mesh.geometry.dim)
        f_p = dolfinx.fem.Constant(mesh, 0.0)
        L_u = dolfinx.fem.form(
            [ufl.inner(f_u, v) * ufl.dx, ufl.inner(f_p, q) * ufl.dx])

        # -- Stokes BCs
        noslip = np.zeros(mesh.geometry.dim, dtype=PETSc.ScalarType)
        noslip_facets = facet_tags.indices[
            facet_tags.values == Labels.plate_wedge]
        bc_plate = dolfinx.fem.dirichletbc(
            noslip, dolfinx.fem.locate_dofs_topological(
                V, mesh.topology.dim-1, noslip_facets), V)

        facets = facet_tags.indices[
            np.isin(facet_tags.values, (Labels.slab_wedge, Labels.slab_plate))]
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

        # Stokes linear system and solver
        self.a_u = a_u
        self.a_p = a_p
        self.L_u = L_u
        self.A_u = dolfinx.fem.petsc.create_matrix_block(a_u)
        self.P_u = dolfinx.fem.petsc.create_matrix_block(a_p)
        self.b_u = dolfinx.fem.petsc.create_vector_block(L_u)

        ksp_u = PETSc.KSP().create(mesh.comm)

        V_map = V.dofmap.index_map
        Q_map = Q.dofmap.index_map
        offset_u = V_map.local_range[0] * V.dofmap.index_map_bs + \
                   Q_map.local_range[0]
        offset_p = offset_u + V_map.size_local * V.dofmap.index_map_bs
        is_u = PETSc.IS().createStride(
            V_map.size_local * V.dofmap.index_map_bs, offset_u, 1,
            comm=PETSc.COMM_SELF)
        is_p = PETSc.IS().createStride(
            Q_map.size_local, offset_p, 1, comm=PETSc.COMM_SELF)
        ksp_u.getPC().setFieldSplitIS(("u", is_u), ("p", is_p))

        if use_iterative_solver:
            ksp_u.setOperators(self.A_u, self.P_u)
            ksp_u.setType("fgmres")
            ksp_u.getPC().setType("fieldsplit")
            ksp_u.getPC().setFieldSplitType(PETSc.PC.CompositeType.MULTIPLICATIVE)
            ksp_u.getPC().setFieldSplitIS(("u", is_u), ("p", is_p))
            ksp_u.setTolerances(rtol=1e-9)

            monitor_n_digits = int(np.ceil(np.log10(ksp_u.max_it)))
            def monitor(ksp, it, r):
                PETSc.Sys.Print(f"{it: {monitor_n_digits}d}: {r:.3e}")

            ksp_u_u, ksp_u_p = ksp_u.getPC().getFieldSplitSubKSP()
            ksp_u.setMonitor(monitor)
            ksp_u_u.setType("preonly")
            ksp_u_u.getPC().setType("hypre")
            ksp_u_p.setType("preonly")
            ksp_u_p.getPC().setType("bjacobi")
        else:
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


class StokesEvolving(Stokes):

    def init(self, uh, Th, eta, slab_velocity, use_iterative_solver):
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

        # Free slip terms
        n = ufl.FacetNormal(mesh)
        h = ufl.CellDiameter(mesh)
        ds = ufl.Measure("ds", subdomain_data=facet_tags)
        alpha = dolfinx.fem.Constant(mesh, 20.0 * V.ufl_element().degree() ** 2)

        # We add the plate surface to free slip BCs assuming a high viscosity
        # is used to model rigidity
        ds_fs = ds((Labels.free_slip, Labels.plate_top))
        a_u00 += - ufl.inner(sigma(u, uh, Th), ufl.outer(ufl.dot(v, n) * n, n)) * ds_fs\
                 - ufl.inner(ufl.outer(ufl.dot(u, n) * n, n), sigma(v, uh, Th)) * ds_fs\
                 + 2 * eta(uh, Th) * alpha / h * ufl.inner(ufl.outer(ufl.dot(u, n) * n, n), ufl.outer(v, n)) * ds_fs
        a_u01 += ufl.inner(p, ufl.dot(v, n)) * ds_fs
        a_u10 += ufl.inner(ufl.dot(u, n), q) * ds_fs

        a_u = dolfinx.fem.form(
            [[a_u00, a_u01],
             [a_u10, None]])
        a_p11 = dolfinx.fem.form(-eta(uh, Th)**-1 * ufl.inner(p, q) * ufl.dx)
        a_p = [[a_u[0][0], a_u[0][1]],
               [a_u[1][0], a_p11]]

        f_u = dolfinx.fem.Constant(mesh, [0.0] * mesh.geometry.dim)
        f_p = dolfinx.fem.Constant(mesh, 0.0)
        L_u = dolfinx.fem.form(
            [ufl.inner(f_u, v) * ufl.dx, ufl.inner(f_p, q) * ufl.dx])

        # -- Stokes BCs
        facets = facet_tags.indices[
            np.isin(facet_tags.values, (Labels.slab_wedge, Labels.slab_plate))]
        assert isinstance(slab_velocity, (np.ndarray, dolfinx.fem.Function))
        if isinstance(slab_velocity, np.ndarray):
            bc_slab = dolfinx.fem.dirichletbc(
                slab_velocity, dolfinx.fem.locate_dofs_topological(
                    V, mesh.topology.dim-1, facets), V)
        elif isinstance(slab_velocity, dolfinx.fem.Function):
            bc_slab = dolfinx.fem.dirichletbc(
                slab_velocity, dolfinx.fem.locate_dofs_topological(
                    V, mesh.topology.dim-1, facets))

        # There is no plate BC, only free slip, as high vsicosity should handle
        # the rigidity.
        self.bcs_u = [bc_slab]

        # Stokes linear system and solver
        self.a_u = a_u
        self.a_p = a_p
        self.L_u = L_u
        self.A_u = dolfinx.fem.petsc.create_matrix_block(a_u)
        self.P_u = dolfinx.fem.petsc.create_matrix_block(a_p)
        self.b_u = dolfinx.fem.petsc.create_vector_block(L_u)

        ksp_u = PETSc.KSP().create(mesh.comm)

        V_map = V.dofmap.index_map
        Q_map = Q.dofmap.index_map
        offset_u = V_map.local_range[0] * V.dofmap.index_map_bs + \
                   Q_map.local_range[0]
        offset_p = offset_u + V_map.size_local * V.dofmap.index_map_bs
        is_u = PETSc.IS().createStride(
            V_map.size_local * V.dofmap.index_map_bs, offset_u, 1,
            comm=PETSc.COMM_SELF)
        is_p = PETSc.IS().createStride(
            Q_map.size_local, offset_p, 1, comm=PETSc.COMM_SELF)
        ksp_u.getPC().setFieldSplitIS(("u", is_u), ("p", is_p))

        if use_iterative_solver:
            ksp_u.setOperators(self.A_u, self.P_u)
            ksp_u.setType("fgmres")
            ksp_u.getPC().setType("fieldsplit")
            ksp_u.getPC().setFieldSplitType(PETSc.PC.CompositeType.MULTIPLICATIVE)
            ksp_u.getPC().setFieldSplitIS(("u", is_u), ("p", is_p))
            ksp_u.setTolerances(rtol=1e-9)

            monitor_n_digits = int(np.ceil(np.log10(ksp_u.max_it)))
            def monitor(ksp, it, r):
                PETSc.Sys.Print(f"{it: {monitor_n_digits}d}: {r:.3e}")

            ksp_u_u, ksp_u_p = ksp_u.getPC().getFieldSplitSubKSP()
            ksp_u.setMonitor(monitor)
            ksp_u_u.setType("preonly")
            ksp_u_u.getPC().setType("hypre")
            ksp_u_p.setType("preonly")
            ksp_u_p.getPC().setType("bjacobi")
        else:
            ksp_u.setOperators(self.A_u)
            ksp_u.setType("preonly")
            pc_u = ksp_u.getPC()
            pc_u.setType("lu")
            pc_u.setFactorSolverType("mumps")

        self.ksp_u = ksp_u
        self.x_u = self.A_u.createVecLeft()


class Heat:

    def __init__(self, mesh, facet_tags, p_order):
        self.mesh = mesh
        self.facet_tags = facet_tags
        self.p_order = p_order
        self.S = dolfinx.fem.FunctionSpace(mesh, ("CG", p_order))

    def init(self, uh, slab_data, depth, use_iterative_solver,
             Th0: dolfinx.fem.Function=None, dt: dolfinx.fem.Constant=None):
        if (Th0 is None) ^ (dt is None):
            raise RuntimeError(
                "Time dependent formulation requires Th0 and dt")

        facet_tags = self.facet_tags
        mesh = self.mesh
        S = self.S

        T = ufl.TrialFunction(S)
        s = ufl.TestFunction(S)

        # Heat system
        a_T = (
            ufl.inner(slab_data.k_prime * ufl.grad(T), ufl.grad(s)) * ufl.dx
            + ufl.inner(ufl.dot(slab_data.rho * slab_data.cp * uh, ufl.grad(T)), s) * ufl.dx
        )
        Q_prime_constant = dolfinx.fem.Constant(mesh, slab_data.Q_prime)
        L_T = ufl.inner(Q_prime_constant, s) * ufl.dx

        if Th0 is not None:
            a_T += ufl.inner(
                slab_data.rho * slab_data.cp * T / dt, s) * ufl.dx
            L_T += ufl.inner(
                slab_data.rho * slab_data.cp * Th0 / dt, s) * ufl.dx

        # Weak heat BCs
        overring_temp = dolfinx.fem.Function(S)
        overring_temp.interpolate(
            lambda x: sz.model.overriding_side_temp(x, slab_data, depth))
        overring_temp.x.scatter_forward()

        ds = ufl.Measure("ds", subdomain_data=facet_tags)
        alpha = dolfinx.fem.Constant(mesh, 20.0 * max(self.p_order, 1)**2)
        h = ufl.CellDiameter(mesh)
        n = ufl.FacetNormal(mesh)
        ds_right = ds((Labels.wedge_right, Labels.plate_right))
        inlet_condition = ufl.conditional(ufl.lt(ufl.dot(uh, n), 0), 1, 0)
        a_T += inlet_condition * (
                ufl.inner(slab_data.k_prime * alpha / h * T, s)
                - ufl.inner(slab_data.k_prime * ufl.grad(T), s * n)
                - ufl.inner(slab_data.k_prime * ufl.grad(s), T * n)
        ) * ds_right
        T_D = overring_temp
        L_T += inlet_condition * (
                ufl.inner(slab_data.k_prime * alpha / h * T_D, s)
                - ufl.inner(slab_data.k_prime * ufl.grad(s), T_D * n)
        ) * ds_right

        a_T, L_T = map(dolfinx.fem.form, (a_T, L_T))

        # Strong heat BCs
        inlet_facets = facet_tags.indices[facet_tags.values == Labels.slab_left]
        inlet_temp = dolfinx.fem.Function(S)
        inlet_temp.interpolate(
            lambda x: sz.model.slab_inlet_temp(x, slab_data, depth))
        inlet_temp.x.scatter_forward()
        bc_inlet = dolfinx.fem.dirichletbc(
            inlet_temp, dolfinx.fem.locate_dofs_topological(
                S, mesh.topology.dim-1, inlet_facets))

        overriding_facets = facet_tags.indices[
            (facet_tags.values == Labels.plate_top) #|
            # (facet_tags.values == Labels.plate_right)
        ]
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
        if use_iterative_solver:
            ksp_T.setType("gmres")
            pc_T = ksp_T.getPC()
            pc_T.setType("bjacobi")
        else:
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