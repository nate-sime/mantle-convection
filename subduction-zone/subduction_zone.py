from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import scipy
import dolfinx
import dolfinx.fem.petsc
import ufl

import mesh_generator
import model

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


# Function spaces for the Stokes problem in the wedge and the temperature
# problem in the full domain
p_order = 2
V = dolfinx.fem.VectorFunctionSpace(wedge_mesh, ("CG", p_order))
Q = dolfinx.fem.FunctionSpace(wedge_mesh, ("CG", p_order-1))
S = dolfinx.fem.FunctionSpace(mesh, ("CG", p_order))

if mesh.comm.rank == 0:
    print(f"Velocity DoFs: "
          f"{V.dofmap.index_map.size_global * V.dofmap.index_map_bs}",
          flush=True)
    print(f"Pressure DoFs: {Q.dofmap.index_map.size_global}", flush=True)
    print(f"Temperature DoFs: {S.dofmap.index_map.size_global}", flush=True)

u, p, T = map(ufl.TrialFunction, (V, Q, S))
v, q, s = map(ufl.TestFunction, (V, Q, S))

uh, ph, Th = map(dolfinx.fem.Function, (V, Q, S))
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

# Interpolate the wedge temperature component for the Stokes problem
S_wedge = dolfinx.fem.FunctionSpace(wedge_mesh, ("CG", p_order))
Th_wedge = dolfinx.fem.Function(S_wedge, name="T_wedge")
Th_to_Th_wedge_interp_data = \
    dolfinx.fem.create_nonmatching_meshes_interpolation_data(
        Th_wedge.function_space.mesh._cpp_object,
        Th_wedge.function_space.element,
        Th.function_space.mesh._cpp_object)
Th_wedge.interpolate(Th, nmm_interpolation_data=Th_to_Th_wedge_interp_data)
Th_wedge.x.scatter_forward()

# Useful initial guess for strainrate dependent viscosities
gkb_wedge_flow_ = lambda x: model.gkb_wedge_flow(x, slab_data.plate_thickness)
uh_full.interpolate(gkb_wedge_flow_, cells=wedge_cells)
uh_full.x.scatter_forward()
uh.interpolate(gkb_wedge_flow_)
uh.x.scatter_forward()

# -- Stokes system
solve_flow = True
if solve_flow:
    eta = model.create_viscosity_1()
    # eta = model.create_viscosity_2a(wedge_mesh)
    # eta = model.create_viscosity_2b(wedge_mesh, slab_data)

    def sigma(u, u0, T):
        return 2 * eta(u0, T) * ufl.sym(ufl.grad(u))


    a_u00 = ufl.inner(sigma(u, uh, Th_wedge), ufl.sym(ufl.grad(v))) * ufl.dx
    a_u01 = - ufl.inner(p, ufl.div(v)) * ufl.dx
    a_u10 = - ufl.inner(q, ufl.div(u)) * ufl.dx
    a_u = dolfinx.fem.form(
        [[a_u00, a_u01],
         [a_u10, None]])
    a_p11 = dolfinx.fem.form(-eta(uh, Th_wedge)**-1 * ufl.inner(p, q) * ufl.dx)
    a_p = [[a_u[0][0], a_u[0][1]],
           [a_u[1][0], a_p11]]

    f_u = dolfinx.fem.Constant(wedge_mesh, [0.0] * mesh.geometry.dim)
    f_p = dolfinx.fem.Constant(wedge_mesh, 0.0)
    L_u = dolfinx.fem.form([ufl.inner(f_u, v) * ufl.dx, ufl.inner(f_p, q) * ufl.dx])

    # -- Stokes BCs
    noslip = np.zeros(mesh.geometry.dim, dtype=PETSc.ScalarType)
    noslip_facets = wedge_facet_tags.indices[
        wedge_facet_tags.values == Labels.plate_wedge]
    bc_plate = dolfinx.fem.dirichletbc(
        noslip, dolfinx.fem.locate_dofs_topological(
            V, mesh.topology.dim-1, noslip_facets), V)

    facets = wedge_facet_tags.indices[wedge_facet_tags.values == Labels.slab_wedge]
    bc_slab = dolfinx.fem.dirichletbc(
        slab_velocity, dolfinx.fem.locate_dofs_topological(
            V, mesh.topology.dim-1, facets), V)

    # The plate BC goes last such that all dofs on the overriding plate
    # have priority and therefore zero flow
    bcs_u = [bc_slab, bc_plate]

    # -- Stokes linear system and solver
    A_u = dolfinx.fem.petsc.create_matrix_block(a_u)
    P_u = dolfinx.fem.petsc.create_matrix_block(a_p)
    b_u = dolfinx.fem.petsc.create_vector_block(L_u)

    def assemble_stokes_system():
        A_u.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix_block(A_u, a_u, bcs=bcs_u)
        A_u.assemble()

        P_u.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix_block(P_u, a_p, bcs=bcs_u)
        P_u.assemble()

        with b_u.localForm() as b_u_local:
            b_u_local.set(0.0)
        dolfinx.fem.petsc.assemble_vector_block(b_u, L_u, a_u, bcs=bcs_u)


    ksp_u = PETSc.KSP().create(mesh.comm)
    ksp_u.setOperators(A_u)
    ksp_u.setType("preonly")
    pc_u = ksp_u.getPC()
    pc_u.setType("lu")
    pc_u.setFactorSolverType("mumps")

    x_u = A_u.createVecLeft()
    def solve_stokes_system():
        ksp_u.solve(b_u, x_u)
        offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
        uh.x.array[:offset] = x_u.array_r[:offset]
        uh.x.scatter_forward()


# -- Heat system
a_T = dolfinx.fem.form(
    ufl.inner(slab_data.k_prime * ufl.grad(T), ufl.grad(s)) * ufl.dx
    + ufl.inner(ufl.dot(slab_data.rho * slab_data.cp * uh_full, ufl.grad(T)), s) * ufl.dx
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

bcs_T = [bc_inlet, bc_overriding]

# -- Heat linear system and solver
A_T = dolfinx.fem.petsc.create_matrix(a_T)
b_T = dolfinx.fem.petsc.create_vector(L_T)

def assemble_temperature_system():
    A_T.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(A_T, a_T, bcs=bcs_T)
    A_T.assemble()

    with b_T.localForm() as b_T_local:
        b_T_local.set(0.0)
    dolfinx.fem.petsc.assemble_vector(b_T, L_T)
    dolfinx.fem.petsc.apply_lifting(b_T, [a_T], bcs=[bcs_T])
    b_T.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b_T, bcs_T)

ksp_T = PETSc.KSP().create(mesh.comm)
ksp_T.setOperators(A_T)
ksp_T.setType("preonly")
pc_T = ksp_T.getPC()
pc_T.setType("lu")
pc_T.setFactorSolverType("mumps")

def solve_temperature_system():
    ksp_T.solve(b_T, Th.vector)
    Th.x.scatter_forward()

# Compare temperature difference between Picard iterations
Th0 = dolfinx.fem.Function(Th.function_space)
Th0.vector.array = Th.vector.array_r
Th0.x.scatter_forward()

for picard_it in range(max_picard_its := 20):
    # Solve Stokes and interpolate velocity approximation into full geometry
    if solve_flow:
        assemble_stokes_system()
        solve_stokes_system()
        uh_full.interpolate(
            uh, cells=wedge_cells,
            nmm_interpolation_data=uh_to_uh_full_interp_data)
        uh_full.x.scatter_forward()

    # Solve temperature
    assemble_temperature_system()
    solve_temperature_system()

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
