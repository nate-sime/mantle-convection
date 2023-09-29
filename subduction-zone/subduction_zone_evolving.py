import pathlib
import json

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import geomdl.exchange
import dolfinx.fem.petsc
import ufl

from sz import model, solvers, interpolation_utils


def print0(*args):
    PETSc.Sys.Print(" ".join(map(str, args)))


sz_data = model.SZData()
Labels = model.Labels


def populate_dg0_marker(marker: dolfinx.fem.Function,
                        cell_tags: dolfinx.mesh.MeshTags):
    """
    Transfer the values of cell tags into a DG0 space. The cell tag values will
    be cast as floating point numbers.

    Args:
        marker: A DG0 space defined on the same mesh as the cell tags
        cell_tags: The cell tags to be transferred

    Returns:
    A DG0 FE function with values transferred from the cell tags
    """
    unique_values = np.unique(cell_tags.values)
    c = dolfinx.fem.Constant(marker.function_space.mesh, 0.0)
    c_expr = dolfinx.fem.Expression(
        c, marker.function_space.element.interpolation_points())
    for val in unique_values:
        c.value = float(val)
        marker.interpolate(c_expr, cell_tags.indices[cell_tags.values == val])
    marker.x.scatter_forward()
    return marker


def solve_slab_problem(
        file_path: pathlib.Path, Th0_mesh0: dolfinx.fem.Function = None,
        dt: float = None,
        slab_spline: geomdl.abstract.Curve | geomdl.abstract.Surface = None,
        slab_spline_m: geomdl.abstract.Curve | geomdl.abstract.Surface = None,
        all_domains_overlap: bool = False):
    """
    Solve the evolving subduction zone model at a given time step.

    Args:
        file_path: Path to the XDMF mesh file
        Th0_mesh0: Previous time step's temperature field defined on the
         previous time step's mesh
        dt: Time step size
        slab_spline: Current time step slab interface spline
        slab_spline_m: Previous time step slab interface spline
        all_domains_overlap: If both the previous and current time step's
         computational domains fully overlap, set to True such that unnecessary
         computation of non-overlapping mesh interpolation may be ingored.

    Returns:
    Subduction zone temperature and velocity data snapshot.
    """
    if ((slab_spline is None) ^ (slab_spline_m is None)
            ^ (Th0_mesh0 is None) ^ (dt is None)):
        raise RuntimeError(
            "Time dependent evolving slab requires all parameters provided")

    # Read meshes and partition over all processes
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, file_path, "r") as fi:
        mesh = fi.read_mesh(
            name="zone", ghost_mode=dolfinx.mesh.GhostMode.none)
        mesh.topology.create_connectivity(mesh.topology.dim - 1, 0)
        facet_tags = fi.read_meshtags(mesh, name="zone_facets")
        cell_tags = fi.read_meshtags(mesh, name="zone_cells")

        wedge_mesh = fi.read_mesh(
            name="wedgeplate", ghost_mode=dolfinx.mesh.GhostMode.none)
        wedge_mesh.topology.create_connectivity(wedge_mesh.topology.dim - 1, 0)
        wedge_facet_tags = fi.read_meshtags(
            wedge_mesh, name="wedgeplate_facets")
        wedge_cell_tags = fi.read_meshtags(
            wedge_mesh, name="wedgeplate_cells")

        slab_mesh = fi.read_mesh(
            name="slab", ghost_mode=dolfinx.mesh.GhostMode.none)
        slab_mesh.topology.create_connectivity(slab_mesh.topology.dim - 1, 0)
        slab_facet_tags = fi.read_meshtags(slab_mesh, name="slab_facets")

    # Set up the Stokes and Heat problems in the full zone and its subdomains
    p_order = 2
    tdim = mesh.topology.dim
    stokes_problem_wedge = solvers.StokesEvolving(
        wedge_mesh, wedge_facet_tags, p_order)
    stokes_problem_slab = solvers.StokesEvolving(
        slab_mesh, slab_facet_tags, p_order)
    heat_problem = solvers.Heat(mesh, cell_tags, facet_tags, p_order)

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
    Th.interpolate(lambda x: np.full_like(x[0], sz_data.Ts))
    Th.name = "T"
    uh_wedge.name = "u"

    # Interpolation data to transfer the velocity on wedge/slab to the full mesh
    V_full = dolfinx.fem.FunctionSpace(
        mesh, ("DG", p_order, (mesh.geometry.dim,)))
    uh_full = dolfinx.fem.Function(V_full, name="u_full")

    wedge_cells = cell_tags.indices[np.isin(cell_tags.values, (Labels.wedge, Labels.plate))]
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
    if use_coupling_depth := True:
        # Ideally a coupling depth is employed such that spurious flow is not
        # initiated in the plate.
        couple_depth = dolfinx.fem.Constant(
            wedge_mesh, np.array(sz_data.plate_thickness, dtype=np.float64))
        full_couple_depth = dolfinx.fem.Constant(
            wedge_mesh, np.array(couple_depth + 10.0, dtype=np.float64))
    else:
        couple_depth, full_couple_depth = None, None

    tau = ufl.as_vector((0, -1) if tdim == 2 else (0, 0, -1))
    wedge_interface_tangent = solvers.steepest_descent(
        stokes_problem_wedge.V, tau, couple_depth=couple_depth,
        full_couple_depth=full_couple_depth, depth=depth)
    slab_tangent_wedge = solvers.facet_local_projection(
        stokes_problem_wedge.V, wedge_facet_tags, Labels.slab_wedge,
        wedge_interface_tangent)

    slab_interface_tangent = solvers.steepest_descent(
        stokes_problem_slab.V, tau)
    slab_tangent_slab = solvers.facet_local_projection(
        stokes_problem_slab.V, slab_facet_tags,
        [Labels.slab_wedge, Labels.slab_plate], slab_interface_tangent)

    if slab_spline_m:
        import sz.spline_util
        sz.spline_util.slab_velocity_kdtree(
            slab_spline, slab_spline_m, slab_tangent_slab,
            slab_facet_tags.indices[np.isin(slab_facet_tags.values, (Labels.slab_wedge, Labels.slab_plate))],
            dt, resolution=256)
        sz.spline_util.slab_velocity_kdtree(
            slab_spline, slab_spline_m, slab_tangent_wedge,
            wedge_facet_tags.indices[np.isin(wedge_facet_tags.values, (Labels.slab_wedge, Labels.slab_plate))],
            dt, resolution=256)

    # Design a viscosity that is discontinuous across the wedge-plate boundary
    eta_wedge_is_linear = True
    eta_wedge_marker = dolfinx.fem.Function(
        dolfinx.fem.FunctionSpace(wedge_mesh, ("DG", 0)))
    populate_dg0_marker(eta_wedge_marker, wedge_cell_tags)
    in_plate = ufl.lt(abs(eta_wedge_marker - Labels.plate), 0.1)
    def eta_wedge(u, T):
        eta_wedge_val = model.create_viscosity_isoviscous()(u, T)
        return ufl.conditional(in_plate, 1e5*eta_wedge_val, eta_wedge_val)

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
        if all_domains_overlap:
            nmm_data = dolfinx.fem.create_nonmatching_meshes_interpolation_data(
                Th0.function_space.mesh._cpp_object,
                Th0.function_space.element,
                Th0_mesh0.function_space.mesh._cpp_object)
            Th0.interpolate(Th0_mesh0, nmm_interpolation_data=nmm_data)
        else:
            interpolation_utils.nonmatching_interpolate(Th0_mesh0, Th0)
        Th0.x.scatter_forward()

    slab_inlet_temp = lambda x: model.slab_inlet_temp(
        x, sz_data, depth, sz_data.age_slab)
    overriding_side_temp = lambda x: sz_data.overriding_side_temp(
        x, sz_data, depth)
    heat_problem.init(
        uh_full, sz_data, depth, slab_inlet_temp, overriding_side_temp,
        use_iterative_solver=False, Th0=Th0, dt=dt)

    # Useful initial guess for strainrate dependent viscosities
    if tdim == 2:
        gkb_wedge_flow_ = lambda x: model.gkb_wedge_flow(
            x, sz_data.plate_thickness)
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


def copy_spline(spline0: geomdl.abstract.Curve | geomdl.abstract.Surface):
    """
    Given a geomdl Curve or Surface, return a copy.

    Args:
        spline0: Spline to copy

    Returns:
    Copied spline
    """
    if spline0.pdimension == 1:
        spline = geomdl.BSpline.Curve()
        spline.degree = spline0.degree
        spline.ctrlpts = spline0.ctrlpts
        spline.knotvector = spline0.knotvector
        return spline
    elif spline0.pdimension == 2:
        spline = geomdl.BSpline.Surface()
        spline.degree = spline0.degree
        spline.ctrlpts_size_u = spline0.ctrlpts_size_u
        spline.ctrlpts_size_v = spline0.ctrlpts_size_v
        spline.ctrlpts = spline0.ctrlpts
        spline.knotvector = spline0.knotvector
        return spline
    raise NotImplementedError("Spline type not known")


def xdmf_interpolator(u):
    """
    Interpolate the FE function in a space compatible with the mesh coordinate
    element. Useful for outputting to XDMF.

    Args:
        u: Function to interpolate

    Returns:
    Interpolated function
    """
    mesh = u.function_space.mesh
    mesh_p = mesh.ufl_domain().ufl_coordinate_element().degree()
    mesh_fam = mesh.ufl_domain().ufl_coordinate_element().family()

    u_mesh = dolfinx.fem.Function(dolfinx.fem.FunctionSpace(
        mesh, (mesh_fam, mesh_p, u.ufl_shape)))
    u_mesh.interpolate(u)
    u_mesh.x.scatter_forward()
    return u_mesh


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

    # Theta-scheme discretised spline deformation states
    slab_spline = copy_spline(slab_spline_t0)
    slab_spline_m = copy_spline(slab_spline_t0)

    def ctrl_pt_transfer_fn(splineA, splineB, theta):
        ctrl0 = np.array(splineA.ctrlpts, dtype=np.float64)
        ctrl1 = np.array(splineB.ctrlpts, dtype=np.float64)
        return (theta * ctrl1 + (1 - theta) * ctrl0).tolist()

    t_yr_steps = np.linspace(0.0, t_final_yr, n_slab_steps + 1)
    dt = sz_data.t_yr_to_ndim(t_yr_steps[1] - t_yr_steps[0])

    for i, t_yr in enumerate(t_yr_steps):
        if i == 0:
            # Initial steady state
            mesh0, T_data0, u_data0 = solve_slab_problem(
                input_directory / f"subduction_zone_{i:{idx_fmt}}.xdmf")
            Th0_mesh0 = T_data0[0]
        else:
            # Parameterised spline in time
            theta = t_yr / t_final_yr
            theta0 = t_yr_steps[i - 1] / t_final_yr
            slab_spline.ctrlpts = ctrl_pt_transfer_fn(
                slab_spline_t0, slab_spline_tfinal, theta)
            slab_spline_m.ctrlpts = ctrl_pt_transfer_fn(
                slab_spline_t0, slab_spline_tfinal, theta0)

            mesh0, T_data0, u_data0 = solve_slab_problem(
                input_directory / f"subduction_zone_{i:{idx_fmt}}.xdmf",
                Th0_mesh0=Th0_mesh0, dt=dt, slab_spline=slab_spline,
                slab_spline_m=slab_spline_m)

        Th0_mesh0 = T_data0[0]
        uh_mesh0 = u_data0[2]
        with dolfinx.io.XDMFFile(
                mesh0.comm,
                output_directory / f"temperature_{i:{idx_fmt}}.xdmf",
                "w") as fi:
            fi.write_mesh(Th0_mesh0.function_space.mesh)
            fi.write_function(xdmf_interpolator(Th0_mesh0), t=t_yr / 1e6)

        with dolfinx.io.XDMFFile(
                mesh0.comm,
                output_directory / f"velocity_{i:{idx_fmt}}.xdmf",
                "w") as fi:
            fi.write_mesh(uh_mesh0.function_space.mesh)
            fi.write_function(xdmf_interpolator(uh_mesh0), t=t_yr / 1e6)
