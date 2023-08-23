from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import scipy
import dolfinx
import dolfinx.fem.petsc
import ufl

import mesh_generator
import model
import solvers


def print0(*args):
    if MPI.COMM_WORLD.rank == 0:
        print(*args, flush=True)


slab_data = model.SlabData()
Labels = mesh_generator.Labels

# Read meshes and partition over all processes
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "subduction_zone.xdmf", "r") as fi:
    mesh = fi.read_mesh(
        name="zone", ghost_mode=dolfinx.cpp.mesh.GhostMode.none)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, 0)
    facet_tags = fi.read_meshtags(mesh, name="zone_facets")
    cell_tags = fi.read_meshtags(mesh, name="zone_cells")

    wedge_mesh = fi.read_mesh(
        name="wedge", ghost_mode=dolfinx.cpp.mesh.GhostMode.none)
    wedge_mesh.topology.create_connectivity(wedge_mesh.topology.dim - 1, 0)
    wedge_facet_tags = fi.read_meshtags(wedge_mesh, name="wedge_facets")

    slab_mesh = fi.read_mesh(
        name="slab", ghost_mode=dolfinx.cpp.mesh.GhostMode.none)
    slab_mesh.topology.create_connectivity(slab_mesh.topology.dim - 1, 0)
    slab_facet_tags = fi.read_meshtags(slab_mesh, name="slab_facets")

# Set up the Stokes and Heat problems in the full zone and its subdomains
p_order = 2
stokes_problem_wedge = solvers.Stokes(wedge_mesh, wedge_facet_tags, p_order)
stokes_problem_slab = solvers.Stokes(slab_mesh, slab_facet_tags, p_order)
heat_problem = solvers.Heat(mesh, facet_tags, p_order)

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
z_hat = ufl.as_vector((0, -1))
slab_tangent_wedge = solvers.tangent_approximation(
    stokes_problem_wedge.V, wedge_facet_tags, Labels.slab_wedge, z_hat,
    y_plate=plate_y, y_couple=couple_y)
slab_tangent_slab = solvers.tangent_approximation(
    stokes_problem_slab.V, slab_facet_tags,
    [Labels.slab_wedge, Labels.slab_plate], z_hat)

eta_wedge = model.create_viscosity_1()
eta_slab = model.create_viscosity_1()
stokes_problem_wedge.init(uh_wedge, Th_wedge, eta_wedge, slab_tangent_wedge)
stokes_problem_slab.init(uh_slab, Th_slab, eta_slab, slab_tangent_slab)
heat_problem.init(uh_full, slab_data)

# Useful initial guess for strainrate dependent viscosities
gkb_wedge_flow_ = lambda x: model.gkb_wedge_flow(x, slab_data.plate_thickness)
uh_full.interpolate(gkb_wedge_flow_, cells=wedge_cells)
uh_full.x.scatter_forward()
uh_wedge.interpolate(gkb_wedge_flow_)
uh_wedge.x.scatter_forward()

# Compare temperature difference between Picard iterations
Th0 = dolfinx.fem.Function(Th.function_space)
Th0.vector.array = Th.vector.array_r
Th0.x.scatter_forward()

solve_flow = True
for picard_it in range(max_picard_its := 25):
    # Solve Stokes and interpolate velocity approximation into full geometry
    if solve_flow:
        print0("Solving wedge problem")
        stokes_problem_wedge.assemble_stokes_system()
        stokes_problem_wedge.solve_stokes_system(uh_wedge)

        print0("Solving slab problem")
        stokes_problem_slab.assemble_stokes_system()
        stokes_problem_slab.solve_stokes_system(uh_slab)

        uh_full.interpolate(
            uh_slab, cells=slab_cells,
            nmm_interpolation_data=uh_slab2full_interp_data)
        uh_full.interpolate(
            uh_wedge, cells=wedge_cells,
            nmm_interpolation_data=uh_wedge2full_interp_data)
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

    if T_diff < (picard_tol := 1e-6):
        break

    # Interpolate temperature approximation into wedge geometry
    Th_wedge.interpolate(Th, nmm_interpolation_data=Th_full2wedge_interp_data)
    Th_wedge.x.scatter_forward()
    Th_slab.interpolate(Th, nmm_interpolation_data=Th_full2slab_interp_data)
    Th_slab.x.scatter_forward()
    Th0.vector.array = Th.vector.array_r
    Th0.x.scatter_forward()

tree = dolfinx.geometry.bb_tree(mesh, mesh.geometry.dim)
points6060 = np.array([60.0, -60.0, 0], np.float64)
cell_candidates = dolfinx.geometry.compute_collisions_points(tree, points6060)
cell_collided = dolfinx.geometry.compute_colliding_cells(
    mesh, cell_candidates, points6060)

T_6060 = None
if len(cell_collided) > 0:
    T_6060 = Th.eval(
        points6060, cell_collided[0])[0] - slab_data.Ts
T_6060 = mesh.comm.gather(T_6060, root=0)

if mesh.comm.rank == 0:
    print(f"T_6060 = {[T_val for T_val in T_6060 if T_val is not None]}",
          flush=True)

with dolfinx.io.VTXWriter(mesh.comm, "temperature.bp", Th, "bp4") as f:
    f.write(0.0)
with dolfinx.io.VTXWriter(mesh.comm, "velocity_wedge.bp", uh_wedge, "bp4") as f:
    f.write(0.0)
with dolfinx.io.VTXWriter(mesh.comm, "velocity_slab.bp", uh_slab, "bp4") as f:
    f.write(0.0)
with dolfinx.io.VTXWriter(mesh.comm, "velocity.bp", uh_full, "bp4") as f:
    f.write(0.0)
