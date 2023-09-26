from dolfinx import fem, io, cpp
from dolfinx.fem import petsc
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from scipy.special import sph_harm
import ufl
from ufl import (
    dot, inner, outer, div, grad, sym,
    TrialFunction, TestFunction, FacetNormal
)

import mesh_generator


def run_mantle_convection():
    # We define some simulation parameters
    r0 = 1.208318891
    r1 = r0 + 1.0
    mu = 1
    k = 2  # Polynomial degree

    msh, mt, boundary_id = mesh_generator.generate_sectant_cap(
        MPI.COMM_WORLD, r0, r1, n=[12, 12, 12], order=k)

    # We force the quadrature degree here otherwise the formulation of ghat will
    # generate too many points
    metadata = {"quadrature_degree": 4*k + 1}
    dx = ufl.dx(metadata=metadata)
    dS = ufl.dS(metadata=metadata)
    ds = ufl.ds(subdomain_data=mt, metadata=metadata)

    # Function space for the velocity, pressure and temperature
    V = fem.VectorFunctionSpace(msh, ("CG", k))
    Q = fem.FunctionSpace(msh, ("CG", k-1))
    S = fem.FunctionSpace(msh, ("CG", k))


    total_dofs = lambda V: msh.comm.allreduce(
        V.dofmap.index_map.size_local * V.dofmap.index_map_bs, op=MPI.SUM)
    PETSc.Sys.Print(
        f"Total global DoFs (V, Q, S): {tuple(map(total_dofs, (V, Q, S)))}")

    # Define trial and test functions
    u, v = TrialFunction(V), TestFunction(V)
    p, q = TrialFunction(Q), TestFunction(Q)
    T, w = TrialFunction(S), TestFunction(S)

    # FE solutions
    u_h, p_h = fem.Function(V, name="u"), fem.Function(Q, name="p")
    u_spd = fem.Function(S, name="|u|")
    T_n = fem.Function(S, name="T")

    delta_t = fem.Constant(msh, PETSc.ScalarType(1e-3))

    # Assemble cell measure into piecewise constant space
    h = fem.Function(fem.FunctionSpace(msh, ("DG", 0)))
    h.vector.array[:] = cpp.mesh.h(
        msh._cpp_object, msh.topology.dim, np.arange(
            msh.topology.index_map(msh.topology.dim).size_local, dtype=np.int32))

    # Compute the smallest cell size to estimate CFL criterion
    hmin = h.vector.min()[1]

    # Free slip Stokes formulation with interior penalty parameter, alpha
    n = FacetNormal(msh)
    alpha = fem.Constant(msh, PETSc.ScalarType(20.0 * k**2))

    a_00 = (
            inner(2 * mu * sym(grad(u)), sym(grad(v))) * dx
            - inner(2 * mu * sym(grad(u)), outer(dot(v, n) * n, n)) * ds
            - inner(outer(dot(u, n) * n, n), 2 * mu * sym(grad(v))) * ds
            + 2 * mu * alpha / h * inner(outer(dot(u, n) * n, n), outer(v, n)) * ds)

    a_01 = - inner(p, div(v)) * dx + inner(p, dot(v, n)) * ds
    a_10 = - inner(div(u), q) * dx + inner(dot(u, n), q) * ds

    # UFL formulation of radial direction of gravity
    x_ufl = ufl.SpatialCoordinate(msh)
    theta = ufl.atan_2(ufl.sqrt(x_ufl[0]**2 + x_ufl[1]**2), x_ufl[2])
    phi = ufl.atan_2(x_ufl[1], x_ufl[0])
    g_hat = -ufl.as_vector((
            ufl.sin(theta) * ufl.cos(phi),
            ufl.sin(theta) * ufl.sin(phi),
            ufl.cos(theta)))

    # Thermal buoyancy formulation with Rayleigh number
    Ra = fem.Constant(msh, PETSc.ScalarType(1e4))
    L_0 = inner(- Ra * T_n * g_hat, v) * dx
    L_1 = inner(fem.Constant(msh, PETSc.ScalarType(0.0)), q) * dx

    # Block system formulations
    a = fem.form([[a_00, a_01],
                  [a_10, None]])
    L = fem.form([L_0, L_1])

    # Assemble Stokes system
    A = fem.petsc.assemble_matrix_block(a)
    A.assemble()
    b = fem.petsc.assemble_vector_block(L, a)

    # Stokes preconditioner
    p_form = fem.form([[a_00, a_01],
                       [a_10, - mu**-1 * inner(p, q) * dx]])
    P = fem.petsc.assemble_matrix_block(p_form)
    P.assemble()

    # Create index sets
    V_map = V.dofmap.index_map
    Q_map = Q.dofmap.index_map
    offset_u = V_map.local_range[0] * V.dofmap.index_map_bs + Q_map.local_range[0]
    offset_p = offset_u + V_map.size_local * V.dofmap.index_map_bs
    is_u = PETSc.IS().createStride(V_map.size_local * V.dofmap.index_map_bs,
                                   offset_u, 1, comm=PETSc.COMM_SELF)
    is_p = PETSc.IS().createStride(Q_map.size_local,
                                   offset_p, 1, comm=PETSc.COMM_SELF)

    # Create KSP solver
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOptionsPrefix("stokes_")
    ksp.setOperators(A, P)
    ksp.getPC().setType("fieldsplit")
    ksp.getPC().setFieldSplitIS(("u", is_u), ("p", is_p))

    ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()

    opts = PETSc.Options()
    opts["stokes_ksp_type"] = "fgmres"
    opts["stokes_ksp_monitor"] = None
    opts["stokes_ksp_rtol"] =  1e-8
    opts["stokes_pc_type"] = "fieldsplit"
    opts["stokes_pc_fieldsplit_type"] = "schur"
    opts["stokes_pc_fieldsplit_schur_fact_type"] = "full"
    opts["stokes_fieldsplit_u_ksp_type"] = "preonly"
    opts["stokes_fieldsplit_u_pc_type"] = "gamg"
    opts["stokes_fieldsplit_u_pc_gamg_type"] = "agg"
    opts["stokes_fieldsplit_p_ksp_type"] = "preonly"
    opts["stokes_fieldsplit_p_pc_type"] = "bjacobi"
    ksp.setFromOptions()
    ksp.getPC().setUp()
    ksp_u.getPC().getOperators()[0].setBlockSize(V.dofmap.index_map_bs)

    # Define a domain average of the pressure to be used in the pressure correction
    vol = msh.comm.allreduce(
        fem.assemble_scalar(fem.form(
            fem.Constant(msh, PETSc.ScalarType(1.0)) * dx)), op=MPI.SUM)


    def domain_average(msh, v):
        """Compute the average of a function over the domain"""
        return 1 / vol * msh.comm.allreduce(
            fem.assemble_scalar(fem.form(v * dx)), op=MPI.SUM)


    # Solution vector split over blocks
    x = A.createVecRight()
    offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    u_h.x.array[:offset] = x.array_r[:offset]
    u_h.x.scatter_forward()
    p_h.x.array[:(len(x.array_r) - offset)] = x.array_r[offset:]
    p_h.x.scatter_forward()
    p_h.x.array[:] -= domain_average(msh, p_h)

    # Write initial condition to file
    t = 0.0

    # Heat equation formulation
    a_T = inner(grad(T), grad(w)) * dx + inner(dot(u_h, grad(T)), w) * dx
    L_T = inner(fem.Constant(msh, 0.0), w) * dx

    # Heat equation strongly enforced BCs
    dof_finder = lambda label: fem.locate_dofs_topological(
        S, msh.topology.dim - 1, mt.indices[mt.values == boundary_id[label]])
    bcs_T = [
        fem.dirichletbc(fem.Constant(msh, 1.0), dof_finder("core"), S),
        fem.dirichletbc(fem.Constant(msh, 0.0), dof_finder("surface"), S)]

    # Heat equation solver
    ksp_T = PETSc.KSP().create(msh.comm)
    ksp_T.setOptionsPrefix("T_")

    opts_T = PETSc.Options()
    opts_T["T_ksp_monitor"] = None
    opts_T["T_ksp_rtol"] =  1e-8
    opts_T["T_ksp_type"] = "gmres"
    opts_T["T_pc_type"] = "bjacobi"
    ksp_T.setFromOptions()


    def T_initial(x):
        l, m = 3, 2
        eps_s, eps_c = 0.01, 0.01
        r = np.sqrt(np.sum(x**2, axis=0))
        theta = np.arccos(x[2]/r)     # colatitude
        phi = np.arctan2(x[1], x[0])  # longitude
        p_lm = sph_harm(m, l, 0.0, theta).real

        radial = r0 * (r - r1) / (r * (r0 - r1))
        azimuthal = (eps_c * np.cos(m * phi) + eps_s * np.sin(m * phi))
        axial = p_lm * np.sin(np.pi * (r - r0) / (r1 - r0))
        return radial + azimuthal * axial


    T_n.interpolate(T_initial)
    T_n.x.scatter_forward()

    a_T = fem.form(a_T)
    L_T = fem.form(L_T)
    A_T = fem.petsc.create_matrix(a_T)
    b_T = fem.petsc.create_vector(L_T)
    ksp_T.setOperators(A_T)

    # Functional of interest: Nusselt number
    surface_flux_form = fem.form(-dot(grad(T_n), n) * ds(boundary_id["surface"]))
    core_flux_form = fem.form(dot(grad(T_n), n) * ds(boundary_id["core"]))
    surf_area = msh.comm.allreduce(
        fem.assemble_scalar(
            fem.form(1 * ds(boundary_id["surface"], domain=msh))),
        op=MPI.SUM)
    core_area = msh.comm.allreduce(
        fem.assemble_scalar(
            fem.form(1 * ds(boundary_id["core"], domain=msh))),
        op=MPI.SUM)

    def compute_nusselt():
        surf_flux = msh.comm.allreduce(
            fem.assemble_scalar(surface_flux_form), op=MPI.SUM)
        core_flux = msh.comm.allreduce(
            fem.assemble_scalar(core_flux_form), op=MPI.SUM)
        Nu_t = r1 * (r1 - r0) / r0 * surf_flux / surf_area
        Nu_b = r0 * (r1 - r0) / r1 * core_flux / core_area
        return Nu_t, Nu_b


    # Parameters used for CFL criterion estimate
    u_spd_expr = fem.Expression(
        ufl.sqrt(u_h**2), u_spd.function_space.element.interpolation_points())
    fs_norm_form = fem.form(dot(u_h, n)**2 * ds)

    with io.VTXWriter(msh.comm, f"output.bp", [u_h, T_n]) as writer:
        Nu_t_old = -1.0
        for n in range(num_time_steps := 200):
            PETSc.Sys.Print(
                f"step {n} of {num_time_steps}, dt = {delta_t.value:.3e}")
            t += delta_t.value

            with b.localForm() as b_loc:
                b_loc.set(0)
            fem.petsc.assemble_vector_block(b, L, a)

            # The Stokes operator need only be constructed once
            if n > 0:
                ksp.getPC().setReusePreconditioner(True)

            # Compute and scatter Stokes solution
            ksp.solve(b, x)
            u_h.x.array[:offset] = x.array_r[:offset]
            u_h.x.scatter_forward()
            p_h.x.array[:(len(x.array_r) - offset)] = x.array_r[offset:]
            p_h.x.scatter_forward()
            p_h.x.array[:] -= domain_average(msh, p_h)

            # Update dt according to CFL criterion estimate
            u_spd.interpolate(u_spd_expr)
            delta_t.value = (c_cfl := 2.0) * hmin / u_spd.vector.max()[1]

            # Compute heat equation solution
            A_T.zeroEntries()
            fem.petsc.assemble_matrix(A_T, a_T, bcs=bcs_T)
            A_T.assemble()

            with b_T.localForm() as b_T_loc:
                b_T_loc.set(0)
            fem.petsc.assemble_vector(b_T, L_T)
            fem.apply_lifting(b_T, [a_T], bcs=[bcs_T])
            b_T.ghostUpdate(
                addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            fem.set_bc(b_T, bcs_T)

            ksp_T.solve(b_T, T_n.vector)
            T_n.x.scatter_forward()

            Nu_t, Nu_b = compute_nusselt()
            fs_norm = msh.comm.allreduce(
                fem.assemble_scalar(fs_norm_form),
                op=MPI.SUM) ** 0.5
            Nu_t_diff = abs(Nu_t - Nu_t_old) / abs(Nu_t_old)
            PETSc.Sys.Print(f"t = {t:.3e}, "
                            f"Nu_t = {Nu_t:.3e}, Nu_b = {Nu_b:.3e}, "
                            f"<T> = {domain_average(msh, T_n):.3e}, "
                            f"<u> = {domain_average(msh, dot(u_h, u_h))**0.5:.3e}, "
                            f"|u.n| = {fs_norm}, "
                            f"|Nu-Nu0|/Nu0 = {Nu_t_diff:.3e}")

            writer.write(float(n))

            if Nu_t_diff < 1e-6:
                PETSc.Sys.Print(f"Converged.")
                break
            Nu_t_old = Nu_t


if __name__ == "__main__":
    run_mantle_convection()
