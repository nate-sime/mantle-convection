import pandas
import numpy as np
import dolfinx
import dolfinx.fem.petsc
import ufl
import dolfin_dg
from mpi4py import MPI
from petsc4py import PETSc
import enum
import febug

import tb.utils
import tb.stokes
import tb.heat


def gen_Ra_val(case):
    if isinstance(case, Blankenbach):
        if case in (Blankenbach.one_a, Blankenbach.two_a):
            return 1e4
        elif case is Blankenbach.one_b:
            return 1e5
        elif case is Blankenbach.one_c:
            return 1e6
    elif isinstance(case, Tosi):
        return 1e2


def form_mu(T, u, case):
    mesh = T.ufl_domain()
    if isinstance(case, Blankenbach):
        if case in (Blankenbach.one_a, Blankenbach.one_b, Blankenbach.one_c):
            mu = dolfinx.fem.Constant(mesh, 1.0)
        elif case is Blankenbach.two_a:
            mu = ufl.exp(-ufl.ln(1000.0) * T)
    elif isinstance(case, Tosi):
        x = ufl.SpatialCoordinate(mesh)
        z = 1 - x[1]
        gamma_T = dolfinx.fem.Constant(mesh, np.log(1e5))
        gamma_z = dolfinx.fem.Constant(mesh, np.log(1.0))
        mu_lin = ufl.exp(-gamma_T * T + gamma_z * z)

        eta_star = dolfinx.fem.Constant(mesh, 1e-3)
        sigma_y = dolfinx.fem.Constant(mesh, 1.0)
        epsII = ufl.sqrt(ufl.sym(ufl.grad(u))**2)
        mu_plast = eta_star + sigma_y / epsII

        if case is Tosi.tosi_1:
            gamma_T.value = np.log(1e5)
            gamma_z.value = np.log(1.0)
            mu = mu_lin
        elif case is Tosi.tosi_2:
            gamma_T.value = np.log(1e5)
            gamma_z.value = np.log(1.0)
            eta_star.value = 1e-3
            sigma_y.value = 1.0
            mu = 2 * (mu_lin**-1 + mu_plast**-1)**-1
        elif case is Tosi.tosi_3:
            gamma_T.value = np.log(1e5)
            gamma_z.value = np.log(10.0)
            mu = mu_lin
        elif case is Tosi.tosi_4:
            gamma_T.value = np.log(1e5)
            gamma_z.value = np.log(10.0)
            eta_star.value = 1e-3
            sigma_y.value = 1.0
            mu = 2 * (mu_lin**-1 + mu_plast**-1)**-1
    return mu


class Formulation(enum.Enum):
    taylor_hood = enum.auto()
    grad_curl_sipg = enum.auto()
    grad_curl_ripg = enum.auto()



class Blankenbach(enum.Enum):
    one_a = enum.auto()
    one_b = enum.auto()
    one_c = enum.auto()
    two_a = enum.auto()


class Tosi(enum.Enum):
    tosi_1 = enum.auto()
    tosi_2 = enum.auto()
    tosi_3 = enum.auto()
    tosi_4 = enum.auto()


def run_model(p, formulator_class, case, n_ele):
    ra_val = gen_Ra_val(case)
    cell_type = dolfinx.mesh.CellType.triangle

    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, [np.array((0, 0)), np.array((1.0, 1.0))],
        [n_ele, n_ele],
        cell_type=cell_type,
        ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
        diagonal=dolfinx.mesh.DiagonalType.left_right)

    PETSc.Sys.Print(f"Running p={p}, n={n_ele}")
    Ra = dolfinx.fem.Constant(mesh, ra_val)
    n = ufl.FacetNormal(mesh)

    formulator = formulator_class(mesh, p)
    Umid = formulator.create_soln_var()
    U = formulator.create_soln_var()
    U_n = formulator.create_soln_var()
    extra_opts = {}

    formulator_heat = tb.heat.ContinuousLagrange(mesh, p)
    T = formulator_heat.create_soln_var()
    Tmid = formulator_heat.create_soln_var()

    # Finite element residual
    A = 0.05
    T.interpolate(
        lambda x: 1.0 - x[1] + A*np.cos(np.pi*x[0]/1.0)*np.sin(np.pi*x[1]/1.0))

    umid = formulator.velocity(Umid)
    mu = form_mu(Tmid, umid, case)
    f = Ra * Tmid * ufl.as_vector((0, 1))

    F = formulator.formulate(mu, f, U)
    J = ufl.derivative(F, U)

    T_bc = formulator_heat.create_bcs(formulator_heat.function_space)
    F_T = formulator_heat.formulate(
        T, ufl.TestFunction(formulator_heat.function_space), umid)
    J_T = ufl.derivative(F_T, T)

    # -- Picard forms and solvers
    F, J, F_T, J_T = map(dolfinx.fem.form, (F, J, F_T, J_T))
    u_bcs = formulator.create_bcs(formulator.function_space)
    problem_stokes = tb.utils.NonlinearPDE_SNESProblem(F, J, U, u_bcs)
    solver_stokes = tb.utils.create_snes(
        problem_stokes, F, J, max_it=1, extra_opts=extra_opts)

    problem_T = tb.utils.NonlinearPDE_SNESProblem(F_T, J_T, T, T_bc)
    solver_T = tb.utils.create_snes(problem_T, F_T, J_T, max_it=1)

    # -- Solve the picard system
    Tmid.x.array[:] = T.x.array
    solver_stokes.solve(None, U.vector)
    Umid.x.array[:] = U.x.array

    solver_T.solve(None, T.vector)
    T_n = formulator_heat.create_soln_var()
    T_n.x.array[:] = Tmid.x.array

    for j in range(max_it := 100):
        Tmid.x.array[:] = 0.5 * (T.x.array + T_n.x.array)
        U_n.x.array[:] = U.x.array[:]
        solver_stokes.solve(None, U.vector)
        Umid.x.array[:] = 0.5 * (U.x.array + U_n.x.array)

        T_n.x.array[:] = T.x.array
        solver_T.solve(None, T.vector)

        rel_diff = (T.vector - T_n.vector).norm() / T_n.vector.norm()
        PETSc.Sys.Print(
            f"Picard it: {j+1}, relative difference {rel_diff:.3e}")
        if rel_diff < 1e-6:
            PETSc.Sys.Print(f"Picard iteration converged")
            break

    # -- Compute functionals
    TOP, BOTTOM, LEFT, RIGHT = 1, 2, 3, 4

    facets_bot = dolfinx.mesh.locate_entities_boundary(
        mesh, dim=mesh.topology.dim-1,
        marker=lambda x: np.isclose(x[1], 0.0))
    facets_top = dolfinx.mesh.locate_entities_boundary(
        mesh, dim=mesh.topology.dim-1,
        marker=lambda x: np.isclose(x[1], 1.0))
    facet_indices = np.concatenate((facets_top, facets_bot))
    facet_values = np.concatenate((np.full_like(facets_top, TOP),
                                   np.full_like(facets_bot, BOTTOM)))
    asort = np.argsort(facet_indices)

    def assemble_scalar_form(functional):
        err = mesh.comm.allreduce(
            dolfinx.fem.assemble.assemble_scalar(
                dolfinx.fem.form(functional)), op=MPI.SUM)
        return err

    def count_dofs(V):
        return V.dofmap.index_map.size_global * V.dofmap.index_map_bs

    h_measure = dolfinx.cpp.mesh.h(
        mesh._cpp_object, 2, np.arange(
            mesh.topology.index_map(2).size_local, dtype=np.int32))
    hmin = mesh.comm.allreduce(h_measure.min(), op=MPI.MIN)
    hmax = mesh.comm.allreduce(h_measure.max(), op=MPI.MAX)

    ff = dolfinx.mesh.meshtags(mesh, mesh.topology.dim-1, facet_indices[asort], facet_values[asort])
    ds = ufl.Measure("ds", subdomain_data=ff)

    Nu_form_numerator = dolfinx.fem.form(ufl.dot(ufl.grad(T), n) * ds(TOP))
    Nu_form_denominator = dolfinx.fem.form(T * ds(BOTTOM))

    Nu = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(Nu_form_numerator), MPI.SUM) / \
         mesh.comm.allreduce(dolfinx.fem.assemble_scalar(Nu_form_denominator), MPI.SUM)
    Nu = -Nu

    u = formulator.velocity(U)
    eps_u = ufl.sym(ufl.grad(u))
    tau = 2 * mu * ufl.sym(ufl.grad(u))
    T_avg = assemble_scalar_form(T * ufl.dx)
    W_avg = assemble_scalar_form(T * u[1] * ufl.dx)
    Phi_Ra_avg = assemble_scalar_form(ufl.inner(tau, eps_u) / Ra * ufl.dx)
    data = pandas.DataFrame({
        "N": n_ele,
        "hmin": hmin,
        "hmax": hmax,
        "total DoF": count_dofs(formulator.function_space),
        "<T>": T_avg,
        "Nu": Nu,
        "u_rms": assemble_scalar_form(u**2 * ufl.dx)**0.5,
        "div(vel) L2": assemble_scalar_form(ufl.div(u) ** 2 * ufl.dx)**0.5,
        "<W>": W_avg,
        "<Phi> / Ra": Phi_Ra_avg,
        "delta": abs(W_avg - Phi_Ra_avg) / max(W_avg, Phi_Ra_avg)
    }, index=[0])

        # Output function checkpoint for particle advection plot
    if output_velcoity := False:
        import adios4dolfinx
        finame = f"./checkpoint/uh_{case}_{formulation}_p{p}_n{n_ele}.bp"
        adios4dolfinx.write_mesh(mesh, finame)
        uh = formulator.velocity_for_output(U)
        adios4dolfinx.write_function(uh, finame)

    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--distribute", type=bool, default=False)
    parser.add_argument("--case", type=str, default="Tosi.tosi_1")
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--form", type=str, default="grad_curl")

    args = parser.parse_args()

    p = 2
    formulator_class = tb.stokes.TaylorHood
    # formulator_class = tb.stokes.C0_SIPG
    case = Blankenbach.one_a
    df = pandas.DataFrame()
    for n_ele in [8, 16, 32]:
        data = run_model(p, formulator_class, case, n_ele)
        df = pandas.concat((df, data), ignore_index=True)

    pandas.set_option('display.max_rows', df.shape[0])
    pandas.set_option('display.max_columns', df.shape[1])
    pandas.set_option('display.width', 180)
    PETSc.Sys.Print(df)
