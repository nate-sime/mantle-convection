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


# Set up the Stokes and Heat problems in the full zone and its subdomains
p_order = 2
stokes_problem = solvers.Stokes(wedge_mesh, wedge_facet_tags, p_order)
heat_problem = solvers.Heat(mesh, facet_tags, p_order)

if mesh.comm.rank == 0:
    print(f"Velocity DoFs: "
          f"{stokes_problem.V.dofmap.index_map.size_global * stokes_problem.V.dofmap.index_map_bs}",
          flush=True)
    print(f"Pressure DoFs: {stokes_problem.Q.dofmap.index_map.size_global}", flush=True)
    print(f"Temperature DoFs: {heat_problem.S.dofmap.index_map.size_global}", flush=True)

# Initialise solution variables
uh = dolfinx.fem.Function(stokes_problem.V)
Th = dolfinx.fem.Function(heat_problem.S)
Th.interpolate(lambda x: np.full_like(x[0], slab_data.Ts))
Th.name = "T"
uh.name = "u"

# Interpolation data to transfer the velocity on wedge to the full mesh
V_full = dolfinx.fem.VectorFunctionSpace(mesh, ("DG", p_order))
uh_full = dolfinx.fem.Function(V_full, name="u_full")
wedge_cells = cell_tags.indices[cell_tags.values == Labels.wedge]
uh_to_uh_full_interp_data = \
    dolfinx.fem.create_nonmatching_meshes_interpolation_data(
        uh_full.function_space.mesh.geometry,
        uh_full.function_space.element,
        uh.function_space.mesh._cpp_object, wedge_cells)

# Set the slab velocity in the full domain velocity approximation
slab_velocity = 2**(-0.5) * np.array([1.0, -1.0], dtype=PETSc.ScalarType)
uh_full.interpolate(lambda x: np.stack([np.full_like(x[i], slab_velocity[i])
                        for i in range(mesh.geometry.dim)]),
                    cells=cell_tags.indices[cell_tags.values == Labels.slab])

# Interpolation data to transfer the full zone temperature to the wedge mesh
S_wedge = dolfinx.fem.FunctionSpace(wedge_mesh, ("CG", p_order))
Th_wedge = dolfinx.fem.Function(S_wedge, name="T_wedge")
Th_to_Th_wedge_interp_data = \
    dolfinx.fem.create_nonmatching_meshes_interpolation_data(
        Th_wedge.function_space.mesh._cpp_object,
        Th_wedge.function_space.element,
        Th.function_space.mesh._cpp_object)
Th_wedge.interpolate(Th, nmm_interpolation_data=Th_to_Th_wedge_interp_data)
Th_wedge.x.scatter_forward()

# Initialise the solvers generating underlying matrices, vectors and KSPs
import dolfinx_mpc.utils
slab_tangent = dolfinx_mpc.utils.facet_normal_approximation(
    stokes_problem.V, wedge_facet_tags, Labels.slab_wedge, tangent=True)

eta = model.create_viscosity_1()
stokes_problem.init(uh, Th_wedge, eta, slab_tangent)
heat_problem.init(uh_full, slab_data)

# Useful initial guess for strainrate dependent viscosities
gkb_wedge_flow_ = lambda x: model.gkb_wedge_flow(x, slab_data.plate_thickness)
uh_full.interpolate(gkb_wedge_flow_, cells=wedge_cells)
uh_full.x.scatter_forward()
uh.interpolate(gkb_wedge_flow_)
uh.x.scatter_forward()

# Compare temperature difference between Picard iterations
Th0 = dolfinx.fem.Function(Th.function_space)
Th0.vector.array = Th.vector.array_r
Th0.x.scatter_forward()

solve_flow = True
for picard_it in range(max_picard_its := 20):
    # Solve Stokes and interpolate velocity approximation into full geometry
    if solve_flow:
        stokes_problem.assemble_stokes_system()
        stokes_problem.solve_stokes_system(uh)
        uh_full.interpolate(
            uh, cells=wedge_cells,
            nmm_interpolation_data=uh_to_uh_full_interp_data)
        uh_full.x.scatter_forward()

    # Solve temperature
    heat_problem.assemble_temperature_system()
    heat_problem.solve_temperature_system(Th)

    # Check convergence
    Th_l2 = Th.vector.norm(PETSc.NormType.NORM_2)
    T_diff = (Th.vector - Th0.vector).norm(PETSc.NormType.NORM_2) / Th_l2

    if mesh.comm.rank == 0:
        print(f"Picard it {picard_it: {int(np.ceil(np.log10(max_picard_its)))}d}: "
              f"T_diff = {T_diff:.3e} ", flush=True)

    if T_diff < (picard_tol := 1e-6):
        break

    # Interpolate temperature approximation into wedge geometry
    Th_wedge.interpolate(Th, nmm_interpolation_data=Th_to_Th_wedge_interp_data)
    Th_wedge.x.scatter_forward()
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
with dolfinx.io.VTXWriter(mesh.comm, "velocity_wedge.bp", uh, "bp4") as f:
    f.write(0.0)
with dolfinx.io.VTXWriter(mesh.comm, "velocity.bp", uh_full, "bp4") as f:
    f.write(0.0)
