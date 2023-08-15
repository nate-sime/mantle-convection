from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import scipy
import dolfinx
import dolfinx.fem.petsc
import ufl

import mesh_generator
import model

model = model.SlabData()
Labels = mesh_generator.Labels
mesh, cell_tags, facet_tags = mesh_generator.generate(MPI.COMM_WORLD)

wedge_mesh, entity_map, _, _ = dolfinx.mesh.create_submesh(
    mesh, mesh.topology.dim,
    cell_tags.indices[cell_tags.values == Labels.wedge])
wedge_facet_tags = mesh_generator.transfer_facet_tags(
    mesh, facet_tags, entity_map, wedge_mesh)

p_order = 2
V = dolfinx.fem.VectorFunctionSpace(wedge_mesh, ("CG", p_order))
Q = dolfinx.fem.FunctionSpace(wedge_mesh, ("CG", p_order-1))
S = dolfinx.fem.FunctionSpace(mesh, ("CG", p_order))

if mesh.comm.rank == 0:
    print(f"Velocity DoFs: "
          f"{V.dofmap.index_map.size_global * V.dofmap.index_map_bs}")
    print(f"Pressure DoFs: {Q.dofmap.index_map.size_global}")
    print(f"Temperature DoFs: {S.dofmap.index_map.size_global}")

u, p, T = map(ufl.TrialFunction, (V, Q, S))
v, q, s = map(ufl.TestFunction, (V, Q, S))

uh, ph, Th = map(dolfinx.fem.Function, (V, Q, S))
Th.interpolate(lambda x: np.full_like(x[0], model.Ts))
Th.name = "T"
uh.name = "u"

# interpolation of the velocity on the full mesh
V_full = dolfinx.fem.VectorFunctionSpace(mesh, ("DG", p_order))
uh_full = dolfinx.fem.Function(V_full, name="u_full")
wedge_cells = cell_tags.indices[cell_tags.values == Labels.wedge]
nm_interp_data = dolfinx.fem.create_nonmatching_meshes_interpolation_data(
        uh_full.function_space.mesh.geometry,
        uh_full.function_space.element,
        uh.function_space.mesh._cpp_object, wedge_cells)

slab_velocity = 2**(-0.5) * np.array([1.0, -1.0], dtype=PETSc.ScalarType)
uh_full.interpolate(lambda x: np.stack([np.full_like(x[i], slab_velocity[i])
                        for i in range(mesh.geometry.dim)]),
                    cells=cell_tags.indices[cell_tags.values == Labels.slab])

def gkb_wedge_flow_(X):
    from numpy import cos, sin, arctan
    plate_thickness = model.plate_thickness
    depth = -plate_thickness - X[1]
    xdist = X[0] - plate_thickness
    xdist[np.isclose(xdist, 0.0)] = 0.000000000000001
    values = np.zeros((2, X.shape[1]), dtype=np.double)
    alfa = arctan(1.0)
    theta = arctan(depth / xdist)
    vtheta = -((alfa - theta) * sin(theta) * sin(alfa) - (
            alfa * theta * sin(alfa - theta))) / (
                     alfa ** 2 - (sin(alfa)) ** 2)
    vr = (((alfa - theta) * cos(theta) * sin(alfa)) - (
            sin(alfa) * sin(theta)) - (alfa * sin(alfa - theta)) + (
                  alfa * theta * cos(alfa - theta))) / (
                 alfa ** 2 - (sin(alfa)) ** 2)
    values[0, :] = - (vtheta * sin(theta) - vr * cos(theta))
    values[1, :] = - (vtheta * cos(theta) + vr * sin(theta))
    values[0, np.isclose(depth, 0.0)] = 0.0
    values[1, np.isclose(depth, 0.0)] = 0.0
    return values

uh_full.interpolate(gkb_wedge_flow_, cells=wedge_cells)
uh_full.x.scatter_forward()

# -- Stokes system
solve_flow = True
if solve_flow:
    def eta(u, T):
        return 1


    def sigma(u, T):
        return 2 * eta(u, T) * ufl.sym(ufl.grad(u))


    a_u00 = ufl.inner(sigma(u, T), ufl.sym(ufl.grad(v))) * ufl.dx
    a_u01 = - ufl.inner(p, ufl.div(v)) * ufl.dx
    a_u10 = - ufl.inner(q, ufl.div(u)) * ufl.dx
    a_u = dolfinx.fem.form(
        [[a_u00, a_u01],
         [a_u10, None]])
    a_p = [[a_u[0][0], a_u[0][1]],
           [a_u[1][0], dolfinx.fem.form(-eta(u, T)**-1 * ufl.inner(p, q) * ufl.dx)]]

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

    # The slab plate BC goes last such that all dofs on the overriding plate
    # have zero flow
    bcs_u = [bc_slab, bc_plate]

    # -- Stokes linear system and solver
    A_u = dolfinx.fem.petsc.assemble_matrix_block(a_u, bcs=bcs_u)
    A_u.assemble()
    P_u = dolfinx.fem.petsc.assemble_matrix_block(a_p, bcs=bcs_u)
    P_u.assemble()
    b_u = dolfinx.fem.petsc.assemble_vector_block(L_u, a_u, bcs=bcs_u)

    ksp_u = PETSc.KSP().create(mesh.comm)
    ksp_u.setOperators(A_u)
    ksp_u.setType("preonly")
    pc_u = ksp_u.getPC()
    pc_u.setType("lu")
    pc_u.setFactorSolverType("mumps")

    x_u = A_u.createVecLeft()
    ksp_u.solve(b_u, x_u)
    offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    uh.x.array[:offset] = x_u.array_r[:offset]
    uh.x.scatter_forward()

    # -- Interpolate velocity approximation into full geometry
    uh_full.interpolate(
        uh, cells=wedge_cells,
        nmm_interpolation_data=nm_interp_data)
    uh_full.x.scatter_forward()

# -- Heat system
a_T = dolfinx.fem.form(
    ufl.inner(model.k_prime * ufl.grad(T), ufl.grad(s)) * ufl.dx
    + ufl.inner(ufl.dot(model.rho * model.cp * uh_full, ufl.grad(T)), s) * ufl.dx
)
Q_prime_constant = dolfinx.fem.Constant(mesh, model.Q_prime)
L_T = dolfinx.fem.form(ufl.inner(Q_prime_constant, s) * ufl.dx)

# -- Heat BCs
def depth(x):
    return -x[1]

def slab_inet_temp_(x):
    vals = np.zeros((1, x.shape[1]))
    Ts = model.Ts
    Twedge_in = model.Twedge_in
    z_scale = model.h_r
    kappa = model.kappa_slab
    t50 = model.t50
    vals[0] = Ts + (Twedge_in - Ts) * scipy.special.erf(
        depth(x) * z_scale / (2.0 * np.sqrt(kappa * t50)))
    return vals

def overriding_side_temp_(x):
    vals = np.zeros((1, x.shape[1]))
    Ts = model.Ts
    T0 = model.Twedge_in
    Zplate = model.plate_thickness
    vals[0] = np.minimum(Ts - (T0 - Ts) / Zplate * (-depth(x)), T0)
    return vals

inlet_facets = facet_tags.indices[facet_tags.values == Labels.slab_left]
inlet_temp = dolfinx.fem.Function(S)
inlet_temp.interpolate(slab_inet_temp_)
inlet_temp.x.scatter_forward()
bc_inlet = dolfinx.fem.dirichletbc(
    inlet_temp, dolfinx.fem.locate_dofs_topological(
        S, mesh.topology.dim-1, inlet_facets))

overriding_facets = facet_tags.indices[
    (facet_tags.values == Labels.plate_top) |
    (facet_tags.values == Labels.plate_right) |
    (facet_tags.values == Labels.wedge_right)]
overring_temp = dolfinx.fem.Function(S)
overring_temp.interpolate(overriding_side_temp_)
overring_temp.x.scatter_forward()
bc_overriding = dolfinx.fem.dirichletbc(
    overring_temp, dolfinx.fem.locate_dofs_topological(
        S, mesh.topology.dim-1, overriding_facets))

bcs_T = [bc_inlet, bc_overriding]

# -- Heat linear system and solver
A_T = dolfinx.fem.petsc.assemble_matrix(a_T, bcs=bcs_T)
A_T.assemble()
b_T = dolfinx.fem.petsc.assemble_vector(L_T)
dolfinx.fem.petsc.apply_lifting(b_T, [a_T], bcs=[bcs_T])
b_T.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
dolfinx.fem.petsc.set_bc(b_T, bcs_T)

ksp_T = PETSc.KSP().create(mesh.comm)
ksp_T.setOperators(A_T)
ksp_T.setType("preonly")
pc_T = ksp_T.getPC()
pc_T.setType("lu")
pc_T.setFactorSolverType("mumps")

ksp_T.solve(b_T, Th.vector)
Th.x.scatter_forward()


tree = dolfinx.geometry.bb_tree(mesh, mesh.geometry.dim)
points6060 = np.array([60.0, -60.0, 0], np.float64)
cell_candidates = dolfinx.geometry.compute_collisions_points(tree, points6060)
cell_collided = dolfinx.geometry.compute_colliding_cells(
    mesh, cell_candidates, points6060)

T_6060 = None
if len(cell_collided) > 0:
    T_6060 = Th.eval(
        points6060, cell_collided[0])[0] - model.Ts
T_6060 = mesh.comm.gather(T_6060, root=0)

if mesh.comm.rank == 0:
    print(f"T_6060 = "
              f"{[T_val for T_val in T_6060 if T_val is not None]}")


with dolfinx.io.VTXWriter(mesh.comm, "temperature.bp", Th, "bp4") as f:
    f.write(0.0)
with dolfinx.io.VTXWriter(mesh.comm, "velocity.bp", uh, "bp4") as f:
    f.write(0.0)
with dolfinx.io.VTXWriter(mesh.comm, "velocity_dg.bp", uh_full, "bp4") as f:
    f.write(0.0)

# import pyvista
# import febug
# plotter = pyvista.Plotter(shape=(1, 2))
# plotter.subplot(0, 0)
# febug.plot_function(uh, plotter=plotter)
# plotter.subplot(0, 1)
# febug.plot_function(Th, plotter=plotter).show()