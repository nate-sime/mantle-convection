import pathlib
import json

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import geomdl.exchange
import dolfinx.fem.petsc
import ufl

from sz import model, solvers


def print0(*args):
    PETSc.Sys.Print(" ".join(map(str, args)))


slab_data = model.SlabData()
Labels = model.Labels

def solve_slab_problem(file_path: pathlib.Path,
                       Th0_mesh0: dolfinx.fem.Function=None,
                       dt: float=None):
    # Read meshes and partition over all processes
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, file_path, "r") as fi:
        mesh = fi.read_mesh(
            name="zone", ghost_mode=dolfinx.mesh.GhostMode.none)
        mesh.topology.create_connectivity(mesh.topology.dim - 1, 0)
        facet_tags = fi.read_meshtags(mesh, name="zone_facets")
        cell_tags = fi.read_meshtags(mesh, name="zone_cells")

        wedge_mesh = fi.read_mesh(
            name="wedge", ghost_mode=dolfinx.mesh.GhostMode.none)
        wedge_mesh.topology.create_connectivity(wedge_mesh.topology.dim - 1, 0)
        wedge_facet_tags = fi.read_meshtags(wedge_mesh, name="wedge_facets")

        slab_mesh = fi.read_mesh(
            name="slab", ghost_mode=dolfinx.mesh.GhostMode.none)
        slab_mesh.topology.create_connectivity(slab_mesh.topology.dim - 1, 0)
        slab_facet_tags = fi.read_meshtags(slab_mesh, name="slab_facets")

    # Set up the Stokes and Heat problems in the full zone and its subdomains
    p_order = 2
    tdim = mesh.topology.dim
    stokes_problem_wedge = solvers.Stokes(wedge_mesh, wedge_facet_tags, p_order)
    stokes_problem_slab = solvers.Stokes(slab_mesh, slab_facet_tags, p_order)
    heat_problem = solvers.Heat(mesh, facet_tags, p_order)

    match tdim:
        case 2:
            def depth(x):
                return -x[1]
        case 3:
            def depth(x):
                return -x[2]

    def stokes_problem_dof_report(problem):
        u_dofs = problem.V.dofmap.index_map.size_global \
                     * problem.V.dofmap.index_map_bs
        p_dofs = problem.Q.dofmap.index_map.size_global
        return f"velocity DoFs: {u_dofs:,} pressure DoFs: {p_dofs:,}"

    print0(f"Wedge {stokes_problem_dof_report(stokes_problem_wedge)}")
    print0(f"Slab {stokes_problem_dof_report(stokes_problem_slab)}")
    print0(f"Temperature DoFs: {heat_problem.S.dofmap.index_map.size_global:,}")

    # Initialise solution variables
    uh_wedge = dolfinx.fem.Function(stokes_problem_wedge.V)
    uh_slab = dolfinx.fem.Function(stokes_problem_slab.V)
    Th = dolfinx.fem.Function(heat_problem.S)
    Th.interpolate(lambda x: np.full_like(x[0], slab_data.Ts))
    Th.name = "T"
    uh_wedge.name = "u"

    # Interpolation data to transfer the velocity on wedge/slab to the full mesh
    V_full = dolfinx.fem.VectorFunctionSpace(mesh, ("DG", p_order))
    uh_full = dolfinx.fem.Function(V_full, name="u_full")

    wedge_cells = cell_tags.indices[cell_tags.values == Labels.wedge]
    uh_wedge2full_interp_data = \
        dolfinx.fem.create_nonmatching_meshes_interpolation_data(
            uh_full.function_space.mesh.geometry,
            uh_full.function_space.element,
            uh_wedge.function_space.mesh._cpp_object, wedge_cells)

    slab_cells = cell_tags.indices[cell_tags.values == Labels.slab]
    uh_slab2full_interp_data = \
        dolfinx.fem.create_nonmatching_meshes_interpolation_data(
            uh_full.function_space.mesh.geometry,
            uh_full.function_space.element,
            uh_slab.function_space.mesh._cpp_object, slab_cells)

    # Interpolation data to transfer the full zone temperature to the wedge mesh
    S_wedge = dolfinx.fem.FunctionSpace(wedge_mesh, ("CG", p_order))
    Th_wedge = dolfinx.fem.Function(S_wedge, name="T_wedge")
    Th_full2wedge_interp_data = \
        dolfinx.fem.create_nonmatching_meshes_interpolation_data(
            Th_wedge.function_space.mesh._cpp_object,
            Th_wedge.function_space.element,
            Th.function_space.mesh._cpp_object)
    Th_wedge.interpolate(Th, nmm_interpolation_data=Th_full2wedge_interp_data)
    Th_wedge.x.scatter_forward()

    S_slab = dolfinx.fem.FunctionSpace(slab_mesh, ("CG", p_order))
    Th_slab = dolfinx.fem.Function(S_slab, name="T_slab")
    Th_full2slab_interp_data = \
        dolfinx.fem.create_nonmatching_meshes_interpolation_data(
            Th_slab.function_space.mesh._cpp_object,
            Th_slab.function_space.element,
            Th.function_space.mesh._cpp_object)
    Th_slab.interpolate(Th, nmm_interpolation_data=Th_full2slab_interp_data)
    Th_slab.x.scatter_forward()

    # Initialise the solvers generating underlying matrices, vectors and KSPs.
    # The tangential velocity approximation for the BCs updates ghosts after
    # solving the projection inside solvers.tangent_approximation
    if use_coupling_depth := False:
        plate_y = dolfinx.fem.Constant(
            wedge_mesh, np.array(-50.0, dtype=np.float64))
        couple_y = dolfinx.fem.Constant(
            wedge_mesh, np.array(plate_y - 10.0, dtype=np.float64))
    else:
        plate_y, couple_y = None, None
    z_hat = ufl.as_vector((0, -1) if tdim == 2 else (0, 0, -1))
    slab_tangent_wedge = solvers.tangent_approximation(
        stokes_problem_wedge.V, wedge_facet_tags, Labels.slab_wedge, z_hat,
        y_plate=plate_y, y_couple=couple_y)
    slab_tangent_slab = solvers.tangent_approximation(
        stokes_problem_slab.V, slab_facet_tags,
        [Labels.slab_wedge, Labels.slab_plate], z_hat)

    eta_wedge_is_linear = True
    eta_wedge = model.create_viscosity_isoviscous()
    eta_slab_is_linear = True
    eta_slab = model.create_viscosity_isoviscous()
    stokes_problem_wedge.init(uh_wedge, Th_wedge, eta_wedge, slab_tangent_wedge,
                              use_iterative_solver=False)
    stokes_problem_slab.init(uh_slab, Th_slab, eta_slab, slab_tangent_slab,
                             use_iterative_solver=False)

    # Interpolate the previous step to the current step's mesh
    Th0 = None
    if Th0_mesh0 is not None:
        Th0 = dolfinx.fem.Function(heat_problem.S)
        nmm_data = dolfinx.fem.create_nonmatching_meshes_interpolation_data(
            Th0.function_space.mesh._cpp_object,
            Th0.function_space.element,
            Th0_mesh0.function_space.mesh._cpp_object)
        Th0.interpolate(Th0_mesh0, nmm_interpolation_data=nmm_data)
        Th0.x.scatter_forward()
    heat_problem.init(uh_full, slab_data, depth, use_iterative_solver=False,
                      Th0=Th0, dt=dt)

    # Useful initial guess for strainrate dependent viscosities
    if tdim == 2:
        gkb_wedge_flow_ = lambda x: model.gkb_wedge_flow(x, slab_data.plate_thickness)
        uh_full.interpolate(gkb_wedge_flow_, cells=wedge_cells)
        uh_full.x.scatter_forward()
        uh_wedge.interpolate(gkb_wedge_flow_)
        uh_wedge.x.scatter_forward()
    else:
        uh_wedge.interpolate(lambda x: (x[0], x[1], x[2]))
        uh_wedge.x.scatter_forward()

    # Compare temperature difference between Picard iterations
    Th0 = dolfinx.fem.Function(Th.function_space)
    Th0.vector.array = Th.vector.array_r
    Th0.x.scatter_forward()

    for picard_it in range(max_picard_its := 25):
        # Solve Stokes and interpolate velocity approximation into full geometry
        if (not eta_wedge_is_linear) or (picard_it == 0):
            print0("Solving wedge problem")
            stokes_problem_wedge.assemble_stokes_system()
            stokes_problem_wedge.solve_stokes_system(uh_wedge)
            uh_full.interpolate(
                uh_wedge, cells=wedge_cells,
                nmm_interpolation_data=uh_wedge2full_interp_data)

        if (not eta_slab_is_linear) or (picard_it == 0):
            print0("Solving slab problem")
            stokes_problem_slab.assemble_stokes_system()
            stokes_problem_slab.solve_stokes_system(uh_slab)
            uh_full.interpolate(
                uh_slab, cells=slab_cells,
                nmm_interpolation_data=uh_slab2full_interp_data)
        uh_full.x.scatter_forward()

        # Solve temperature
        print0("Solving heat problem")
        heat_problem.assemble_temperature_system()
        heat_problem.solve_temperature_system(Th)

        # Check convergence
        Th_l2 = Th.vector.norm(PETSc.NormType.NORM_2)
        T_diff = (Th.vector - Th0.vector).norm(PETSc.NormType.NORM_2) / Th_l2
        print0(f"Picard it {picard_it: {int(np.ceil(np.log10(max_picard_its)))}d}:"
               f" T_diff = {T_diff:.3e}")

        if T_diff < (picard_tol := 1e-4):
            break

        # Interpolate temperature approximation into wedge geometry
        Th_wedge.interpolate(Th, nmm_interpolation_data=Th_full2wedge_interp_data)
        Th_wedge.x.scatter_forward()
        Th_slab.interpolate(Th, nmm_interpolation_data=Th_full2slab_interp_data)
        Th_slab.x.scatter_forward()
        Th0.vector.array = Th.vector.array_r
        Th0.x.scatter_forward()

    T_data = (Th, Th_slab, Th_wedge)
    u_data = (uh_full, uh_slab, uh_wedge)
    return mesh, T_data, u_data


# if __name__ == "__main__":
#     mesh, T_data, u_data = solve_slab_problem(
#         pathlib.Path("subduction_zone.xdmf"))
#     Th = T_data[0]
#     uh_full, uh_slab, uh_wedge = u_data
#
#     tree = dolfinx.geometry.bb_tree(mesh, mesh.geometry.dim)
#     points6060 = np.array([60.0, -60.0, 0], np.float64)
#     cell_candidates = dolfinx.geometry.compute_collisions_points(tree, points6060)
#     cell_collided = dolfinx.geometry.compute_colliding_cells(
#         mesh, cell_candidates, points6060)
#
#     T_6060 = None
#     if len(cell_collided) > 0:
#         T_6060 = Th.eval(
#             points6060, cell_collided[0])[0] - slab_data.Ts
#     T_6060 = mesh.comm.gather(T_6060, root=0)
#
#     if mesh.comm.rank == 0:
#         print(f"T_6060 = {[T_val for T_val in T_6060 if T_val is not None]}",
#               flush=True)
#
#     with dolfinx.io.VTXWriter(mesh.comm, "temperature.bp", Th, "bp4") as f:
#         f.write(0.0)
#     with dolfinx.io.VTXWriter(mesh.comm, "velocity_wedge.bp", uh_wedge, "bp4") as f:
#         f.write(0.0)
#     with dolfinx.io.VTXWriter(mesh.comm, "velocity_slab.bp", uh_slab, "bp4") as f:
#         f.write(0.0)
#     with dolfinx.io.VTXWriter(mesh.comm, "velocity.bp", uh_full, "bp4") as f:
#         f.write(0.0)


if __name__ == "__main__":
    output_directory = pathlib.Path("evolving2d_results")
    input_directory = pathlib.Path("evolving2d")
    with open(input_directory / "metadata.json", "r") as fi:
        meta_data = json.load(fi)
    t_final_yr = meta_data["t_final_yr"]
    n_slab_steps = meta_data["n_slab_steps"]
    idx_fmt = meta_data["idx_fmt"]

    slab_spline_t0 = geomdl.exchange.import_json(
        input_directory / "slab_spline_t0.json")[0]
    slab_spline_tfinal = geomdl.exchange.import_json(
        input_directory / "slab_spline_tfinal.json")[0]

    slab_spline = geomdl.BSpline.Curve()
    slab_spline.degree = slab_spline_t0.degree
    slab_spline.ctrlpts = slab_spline_t0.ctrlpts
    slab_spline.knotvector = slab_spline_t0.knotvector

    def ctrl_pt_transfer_fn(theta):
        ctrl0 = np.array(slab_spline_t0.ctrlpts, dtype=np.float64)
        ctrl1 = np.array(slab_spline_tfinal.ctrlpts, dtype=np.float64)
        return (theta * ctrl1 + (1 - theta) * ctrl0).tolist()

    t_yr_steps = np.linspace(0.0, t_final_yr, n_slab_steps + 1)
    dt = slab_data.t_yr_to_ndim(t_yr_steps[1] - t_yr_steps[0])
    mesh0, T_data0, u_data0 = solve_slab_problem(
        input_directory / f"subduction_zone_{0:{idx_fmt}}.xdmf")
    Th0_mesh0 = T_data0[0]

    def xdmf_interpolator(u):
        mesh = u.function_space.mesh
        mesh_p = mesh.ufl_domain().ufl_coordinate_element().degree()
        mesh_fam = mesh.ufl_domain().ufl_coordinate_element().family()

        u_mesh = dolfinx.fem.Function(dolfinx.fem.FunctionSpace(
            mesh, (mesh_fam, mesh_p)))
        u_mesh.interpolate(u)
        u_mesh.x.scatter_forward()
        return u_mesh

    for i, t_yr in enumerate(t_yr_steps):
        mesh0, T_data0, u_data0 = solve_slab_problem(
            input_directory / f"subduction_zone_{i:{idx_fmt}}.xdmf",
            Th0_mesh0=Th0_mesh0, dt=dt)
        Th0_mesh0 = T_data0[0]
        uh_mesh0 = u_data0[0]
        with dolfinx.io.XDMFFile(
                mesh0.comm,
                output_directory / f"temperature{i:{idx_fmt}}.xdmf",
                "w") as fi:
            fi.write_mesh(Th0_mesh0.function_space.mesh)
            fi.write_function(xdmf_interpolator(Th0_mesh0), t=t_yr / 1e6)
        # quit()
        # with dolfinx.io.VTXWriter(
        #         mesh0.comm, output_directory / f"temperature_{i:{idx_fmt}}.bp",
        #         Th0_mesh0, "bp4") as f:
        #     f.write(t_yr / 1e6)
        # with dolfinx.io.VTXWriter(
        #         mesh0.comm, output_directory / f"velocity_{i:{idx_fmt}}.bp",
        #         uh_mesh0, "bp4") as f:
        #     f.write(t_yr / 1e6)