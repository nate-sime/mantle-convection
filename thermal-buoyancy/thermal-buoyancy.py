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


class ViscosityFormulator:

    def __init__(self, d_eta_T, d_eta_z,
                 eta_star=None, sigma_y=None):
        self.d_eta_T = d_eta_T
        self.d_eta_z = d_eta_z
        self.eta_star = eta_star
        self.sigma_y = sigma_y

    def mu_linear(self, T, u):
        x = ufl.SpatialCoordinate(T.function_space.mesh)
        z = 1 - x[1]
        return ufl.as_ufl(
            ufl.exp(-ufl.ln(self.d_eta_T) * T + ufl.ln(self.d_eta_z) * z))

    def mu_plastic(self, T, u):
        epsII = ufl.sqrt(ufl.sym(ufl.grad(u)) ** 2)
        return self.eta_star + self.sigma_y / epsII

    def mu_combined(self, T, u):
        mu_lin = self.mu_linear(T, u)
        mu_plast = self.mu_plastic(T, u)
        return 2 * (mu_lin**-1 + mu_plast**-1)**-1


cases = {
    "Blankenbach_1a":
        {
            "Ra": 1e4,
            "mu": ViscosityFormulator(d_eta_T=1, d_eta_z=1).mu_linear
        },
    "Blankenbach_2a":
        {
            "Ra": 1e4,
            "mu": ViscosityFormulator(d_eta_T=1e3, d_eta_z=1).mu_linear
        },
    "Tosi_1":
        {
            "Ra": 1e2,
            "mu": ViscosityFormulator(d_eta_T=1e5, d_eta_z=1).mu_linear
        },
    "Tosi_2":
        {
            "Ra": 1e2,
            "mu": ViscosityFormulator(
                d_eta_T=1e5, d_eta_z=1, eta_star=1e-3, sigma_y=1).mu_combined
        },
    "Tosi_3":
        {
            "Ra": 1e2,
            "mu": ViscosityFormulator(d_eta_T=1e5, d_eta_z=10).mu_linear
        },
    "Tosi_4":
        {
            "Ra": 1e2,
            "mu": ViscosityFormulator(
                d_eta_T=1e5, d_eta_z=10, eta_star=1e-3, sigma_y=1).mu_combined
        },
}


class Formulation(enum.Enum):
    taylor_hood = enum.auto()
    grad_curl_sipg = enum.auto()
    grad_curl_ripg = enum.auto()


def run_model(p, formulator_class, case, n_ele, write_solution):
    ra_val = cases[case]["Ra"]
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
    T.x.scatter_forward()

    umid = formulator.velocity(Umid)
    mu = cases[case]["mu"](Tmid, umid)
    f = Ra * Tmid * ufl.as_vector((0, 1))

    F = formulator.formulate(mu, f, U)
    J = ufl.derivative(F, U)

    T_bc = formulator_heat.create_bcs(formulator_heat.function_space)
    F_T = formulator_heat.formulate(
        T, ufl.TestFunction(formulator_heat.function_space), umid)
    J_T = ufl.derivative(F_T, T)

    # Picard forms and solvers
    F, J, F_T, J_T = map(dolfinx.fem.form, (F, J, F_T, J_T))
    u_bcs = formulator.create_bcs(formulator.function_space)

    # Ghost updates are handled by the NonlinearPDE_SNESProblem class
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
            f"Picard it: {j+1}, ‖Tₙ₊₁ − Tₙ‖₂ / ‖Tₙ‖₂ = {rel_diff:.3e}")
        if rel_diff < 1e-6:
            PETSc.Sys.Print(f"Picard iteration converged")
            break

    # Compute functionals
    TOP, BOTTOM = 1, 2

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

    ff = dolfinx.mesh.meshtags(
        mesh, mesh.topology.dim-1, facet_indices[asort], facet_values[asort])
    ds = ufl.Measure("ds", subdomain_data=ff)

    Nu_numerator = assemble_scalar_form(-ufl.dot(ufl.grad(T), n) * ds(TOP))
    Nu_denominator = assemble_scalar_form(T * ds(BOTTOM))
    Nu = Nu_numerator / Nu_denominator

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

    if write_solution:
        import pathlib
        output_dir = pathlib.Path("./results")
        finame_prefix = f"{case}_{formulator_class.__name__}_p{p}_n{n_ele}"
        u_output = formulator.velocity_for_output(U)
        u_output.name = "u"
        T.name = "T"

        temperature_file = output_dir / (finame_prefix + "_temperature.bp")
        with dolfinx.io.VTXWriter(MPI.COMM_WORLD, temperature_file, [T]) as fi:
            fi.write(0.0)

        velocity_file = output_dir / (finame_prefix + "_velocity.bp")
        with dolfinx.io.VTXWriter(
                MPI.COMM_WORLD, velocity_file, [u_output]) as fi:
            fi.write(0.0)

        viscosity_output = dolfinx.fem.Function(dolfinx.fem.FunctionSpace(
            mesh, ("DG", p-2)), name="mu")
        mu_output = cases[case]["mu"](T, u)
        viscosity_output.interpolate(dolfinx.fem.Expression(
            mu_output,
            viscosity_output.function_space.element.interpolation_points(),
            comm=mesh.comm))
        viscosity_file = output_dir / (finame_prefix + "_viscosity.bp")
        with dolfinx.io.VTXWriter(
            MPI.COMM_WORLD, viscosity_file, [viscosity_output]) as fi:
            fi.write(0.0)

        try:
            import adios4dolfinx
            aout = pathlib.Path("./checkpoints/")
            def write_function(f, filename):
                adios4dolfinx.write_mesh(mesh, filename)
                adios4dolfinx.write_function(f, filename)
            write_function(u_output, aout / (finame_prefix + "_velocity.bp"))
            write_function(T, aout / (finame_prefix + "_temperature.bp"))
            write_function(viscosity_output,
                           aout / (finame_prefix + "_viscosity.bp"))
        except ImportError as err:
            PETSc.Sys.Print("adios4dolfinx required to checkpoint solution")

    return data


if __name__ == "__main__":
    # Polynomial degree of discretisation scheme
    p = 3

    # Discretisation scheme
    formulator_class = tb.stokes.C0_SIPG

    # Run all cases, or choose subsets of cases
    example_cases = cases.keys()

    # Number of elements in each direction. For convergence tests use a
    # sequence, e.g. [16, 32, 64]
    n_eles = [16]

    # itera
    for case in example_cases:
        df = pandas.DataFrame()
        for n_ele in n_eles:
            data = run_model(
                p, formulator_class, case, n_ele, write_solution=False)
            df = pandas.concat((df, data), ignore_index=True)

        pandas.set_option('display.max_rows', df.shape[0])
        pandas.set_option('display.max_columns', df.shape[1])
        pandas.set_option('display.width', 180)
        PETSc.Sys.Print(case)
        PETSc.Sys.Print(df)
