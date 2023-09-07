import dolfinx.cpp.io
from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import ufl
import dolfinx
import dolfinx.fem.petsc

import solver_utils
import mesh_generator


Labels = mesh_generator.Labels
dx = ufl.dx(metadata={"quadrature_degree": 4})

def print0(*args):
    PETSc.Sys.Print(", ".join(map(str, args)))


matrix_type = solver_utils.MatrixType.block

# if matrix_type is MatrixType.block:
#     create_vector = dolfin.fem.create_vector_block
#     create_matrix = dolfin.fem.create_matrix_block
# elif matrix_type is MatrixType.nest:
#     create_vector = dolfin.fem.create_vector_nest
#     create_matrix = dolfin.fem.create_matrix_nest

# mesh = XDMFFile(MPI.comm_world, "../meshes/annulus/annulus.xdmf").read_mesh(dolfin.cpp.mesh.GhostMode.none)
# mesh = XDMFFile(MPI.comm_world, "../meshes/circle/circle.xdmf").read_mesh(dolfin.cpp.mesh.GhostMode.none)
# mesh = XDMFFile(MPI.comm_world, "../meshes/cylinder/cylinder.xdmf").read_mesh(dolfin.cpp.mesh.GhostMode.none)
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "disk.xdmf", "r") as fi:
    mesh = fi.read_mesh(
        name="mesh", ghost_mode=dolfinx.cpp.mesh.GhostMode.none)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, 0)
    facet_tags = fi.read_meshtags(mesh, name="facets")

h_on_delta = dolfinx.fem.Constant(mesh, 4.0)
eta_0 = dolfinx.fem.Constant(mesh, 1.0)
phi_0 = dolfinx.fem.Constant(mesh, 0.05)
alpha = dolfinx.fem.Constant(mesh, -27.0)
n = dolfinx.fem.Constant(mesh, 2.0)
nexp = (1 - n) / n
# nexp = Constant(mesh, (1 - n.value) / n.value)
m = dolfinx.fem.Constant(mesh, 1.0)

face_n = ufl.FacetNormal(mesh)

p = 2
CGp = ufl.FiniteElement("CG", mesh.ufl_cell(), p)
CGk = ufl.FiniteElement("CG", mesh.ufl_cell(), p - 1)
CGp_vec = ufl.VectorElement("CG", mesh.ufl_cell(), p)
QC = dolfinx.fem.FunctionSpace(mesh, CGp)
PHI = dolfinx.fem.FunctionSpace(mesh, CGp)
Q = dolfinx.fem.FunctionSpace(mesh, CGk)
V = dolfinx.fem.FunctionSpace(mesh, CGp_vec)

u, p, phi, p_c = map(dolfinx.fem.Function, (V, Q, PHI, QC))
v, q, psi, q_c = map(ufl.TestFunction, (V, Q, PHI, QC))

phi.interpolate(lambda x: 1.0 + 0.01*(0.5 - np.random.rand(x.shape[1])))
ghat = dolfinx.fem.Constant(mesh, [0.0] * mesh.topology.dim)

count_dofs = lambda V: V.dofmap.index_map.size_global * V.dofmap.index_map_bs
print0(f"Problem dim: (Qc: {count_dofs(QC):,} "
       f"Phi: {count_dofs(PHI):,} Q: {count_dofs(Q):,} V: {count_dofs(V):,})")

# u_0_state = (4.0 if mesh.geometry.dim == 3 else 4.0) / r0
# angular_flow_acw_bc = Function(V)
# def flow_acw(x):
#     if mesh.geometry.dim == 2:
#         return u_0_state * np.row_stack((-x[1], x[0]))
#     else:
#         return u_0_state * np.row_stack((-x[2] * x[1], x[2] * x[0], np.zeros_like(x[0])))
# angular_flow_acw_bc.interpolate(flow_acw)

def rot_flow(x: np.ndarray, direction: int):
    cw = np.stack(((-1)**direction * x[1], - (-1)**direction * x[0]))
    cw_norm = np.linalg.norm(cw, axis=0)
    return cw / cw_norm

u_inner = dolfinx.fem.Function(V)
u_inner.interpolate(lambda x: rot_flow(x, 1))
u_outer = dolfinx.fem.Function(V)
u_outer.interpolate(lambda x: 2.0*rot_flow(x, 2))
# u.interpolate(angular_flow_acw_bc)

zero_vector = dolfinx.fem.Function(V)
zero_vector.vector.set(0.0)

facets_inner = facet_tags.indices[facet_tags.values == Labels.inner_face]
bc_top = dolfinx.fem.dirichletbc(
    u_inner, dolfinx.fem.locate_dofs_topological(
        V, mesh.topology.dim - 1, facets_inner))

facets_outer = facet_tags.indices[facet_tags.values == Labels.outer_face]
bc_bottom = dolfinx.fem.dirichletbc(
    u_outer, dolfinx.fem.locate_dofs_topological(
        V, mesh.topology.dim - 1, facets_outer))

bcs_V = [bc_top, bc_bottom]

# if mesh.geometry.dim == 3:
    # bcs = [DirichletBC(V, angular_flow_acw_bc, boundary_top),
    #        DirichletBC(V, angular_flow_acw_bc, boundary_bottom)]
# else:
#     angular_flow_cw_bc = Function(V)
#     flow_cw = lambda x: 0.95 * u_0_state * np.row_stack((x[1], -x[0]))
#     angular_flow_cw_bc.interpolate(flow_cw)
#
#     bcs = [DirichletBC(V, angular_flow_acw_bc, boundary_outer),
#            DirichletBC(V, angular_flow_cw_bc, boundary_inner)]

def eps(u):
    return ufl.sym(ufl.grad(u))

def eta(u, phi):
    eps_II = ufl.sqrt(0.5 * ufl.inner(eps(u), eps(u))
                      + dolfinx.fem.Constant(mesh, 1e-9))
    return eta_0 * ufl.exp(alpha*phi_0*(phi - 1.0)) * eps_II ** nexp

def sigma(u, phi):
    return 2*eta(u, phi)*eps(u)


u_n, p_n, phi_n, p_c_n = map(dolfinx.fem.Function, (V, Q, PHI, QC))

max_t = 6.5
h_measure = dolfinx.cpp.mesh.h(
    mesh._cpp_object, mesh.topology.dim, np.arange(
        mesh.topology.index_map(mesh.topology.dim).size_local, dtype=np.int32))
hmin = mesh.comm.allreduce(h_measure.min(), op=MPI.MIN)
# dt = dolfinx.fem.Constant(mesh, 1.25e-2 * hmin)  # 2.5e-2 for 128x32
dt = dolfinx.fem.Constant(mesh, 1.25e-1 * hmin)  # 2.5e-2 for 128x32
t = dolfinx.fem.Constant(mesh, 0.0)
n_t_steps = int(max_t/dt.value)

# Theta scheme
u_th = 0.5*(u + u_n)
p_th = 0.5*(p + p_n)
phi_th = 0.5*(phi + phi_n)
p_c_th = 0.5*(p_c + p_c_n)

K = phi_th**n
zeta = phi_th**(-m)
xi = eta(u_th, phi_th)*(zeta - 2.0/3.0*phi_0**m)

f1 = (
        (phi - phi_n) * psi * dx
        + dt * (
                ufl.dot(u_th, ufl.grad(phi_th))*psi*dx
                - phi_0**(m-1)*(1 - phi_0*phi_th) * p_c_th / xi * psi * dx
        )
)
f2 = (
        ufl.dot(K * (ufl.grad(p_c_th) + ufl.grad(p_th)), ufl.grad(q_c)) * dx
        + h_on_delta ** 2 * p_c_th / xi * q_c * dx
)
f3 = (
        ufl.inner(sigma(u_th, phi_th), eps(v))*dx
        - p_th*ufl.div(v)*dx
)
f4 = -ufl.div(u_th)*q*dx + phi_0**m*p_c_th/xi*q*dx

F = [f1, f2, f3, f4]
U = [phi, p_c, u, p]
Un = [phi_n, p_c_n, u_n, p_n]
dU = list(map(ufl.TrialFunction, (PHI, QC, V, Q)))
J = solver_utils.derivative_block(F, U, dU)

dp = ufl.TrialFunction(Q)

# P = derivative_nest(F, U, dU)
# P[3][3] = -eta(u_th, phi_th)**-1*dp*q*dx
P = [[J[0][0], J[0][1], J[0][2], J[0][3]],
     [J[1][0], J[1][1], J[1][2], J[1][3]],
     [J[2][0], J[2][1], J[2][2], J[2][3]],
     [J[3][0], J[3][1], J[3][2], -eta(u_th, phi_th)**-1*dp*q*dx]]
# P = [[J[0][0], None, None, None],
#      [None, J[1][1], None, None],
#      [None, None, J[2][2], J[2][3]],
#      [None, None, J[3][2], -eta(u_th, phi_th)**-1*dp*q*dx]]

if matrix_type is solver_utils.MatrixType.block:
    P = None

# nullspace_phi = Function(PHI)
# nullspace_qc = Function(QC)
# nullspace_u = Function(V)
# nullspace_p = Function(Q)
#
# for function in [nullspace_phi, nullspace_qc, nullspace_u]:
#     function.vector.set(0.0)
# nullspace_p.vector.set(1.0)
#
# The pressure nullspace must be reconstructed to match the problem size
# p_nullspc_nest = PETSc.Vec().createNest((nullspace_phi.vector,
#                                          nullspace_qc.vector,
#                                          nullspace_u.vector,
#                                          nullspace_p.vector))
#
# nullspc_nest = cpp.la.VectorSpaceBasis([p_nullspc_nest])
# nullspc_nest.orthonormalize()
#
# rbms = [Function(V) for j in range(mesh.geometry.dim + 1)]
# rbms[0].interpolate(lambda x: np.column_stack((np.ones_like(x[:, 0]), np.zeros_like(x[:, 0]))))
# rbms[1].interpolate(lambda x: np.column_stack((np.zeros_like(x[:, 0]), np.ones_like(x[:, 0]))))
# rbms[2].interpolate(lambda x: np.column_stack((-x[:, 1], x[:, 0])))
#
# rbms = cpp.la.VectorSpaceBasis(list(rbm.vector for rbm in rbms))
# rbms.orthonormalize()

# problem = GenericSNESProblem(J, F, P, bcs, U,
#                              assemble_type=matrix_type,
#                              use_preconditioner=matrix_type is MatrixType.nest)
F, J, P = map(dolfinx.fem.form, (F, J, P))
problem = solver_utils.NonlinearPDE_SNESProblem(F, J, U, bcs_V, P)
problem.F = problem.F_block if matrix_type is solver_utils.MatrixType.block \
    else problem.F_nest
problem.J = problem.J_block if matrix_type is solver_utils.MatrixType.block \
    else problem.J_nest

if matrix_type is solver_utils.MatrixType.block:
    Jmat = dolfinx.fem.petsc.create_matrix_block(J)
    Pmat = None
    Fvec = dolfinx.fem.petsc.create_vector_block(F)
    x0 = dolfinx.fem.petsc.create_vector_block(F)
elif matrix_type is solver_utils.MatrixType.nest:
    Jmat = dolfinx.fem.petsc.create_matrix_nest(J)
    Pmat = dolfinx.fem.petsc.create_matrix_nest(P)
    Fvec = dolfinx.fem.petsc.create_vector_nest(F)
    x0 = dolfinx.fem.petsc.create_vector_nest(F)

if matrix_type is solver_utils.MatrixType.block:
    array_rs = [u_.vector.array_r for u_ in U]
    im_bs = [(u_.function_space.dofmap.index_map,
              u_.function_space.dofmap.index_map_bs)
             for u_ in U]
    dolfinx.cpp.la.petsc.scatter_local_vectors(x0, array_rs, im_bs)
    x0.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
else:
    for u_, sub_vec in zip(U, x0.getNestSubVecs()):
        sub_vec.array[:] = u_.vector.array_r
        sub_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                            mode=PETSc.ScatterMode.FORWARD)

snes = PETSc.SNES().create(mesh.comm)
snes.setTolerances(atol=1e-7, max_it=200)
# snes.getKSP().setTolerances(rtol=1e-1)

if matrix_type is solver_utils.MatrixType.block:
    snes.getKSP().setType("preonly")
    snes.getKSP().getPC().setType("lu")
    snes.getKSP().getPC().setFactorSolverType("mumps")
else:
    nested_IS = Jmat.getNestISs()

    snes.getKSP().setType("fgmres")
    # snes.getKSP().setTolerances(rtol=1e-12)
    snes.getKSP().getPC().setType("fieldsplit")
    snes.getKSP().getPC().setFieldSplitType(PETSc.PC.CompositeType.SYMMETRIC_MULTIPLICATIVE)
    snes.getKSP().getPC().setFieldSplitIS(["phi", nested_IS[0][0]],
                                          ["p_c", nested_IS[0][1]],
                                          ["u", nested_IS[0][2]],
                                          ["p", nested_IS[0][3]])

    ksp_phi, ksp_p_c, ksp_u, ksp_p = snes.getKSP().getPC().getFieldSplitSubKSP()

    ksp_phi.setType("gmres")
    ksp_phi.getPC().setType('hypre')
    # ksp_phi.getPC().setFactorSolverType('mumps')
    ksp_p_c.setType("gmres")
    ksp_p_c.getPC().setType('hypre')
    # ksp_p_c.getPC().setFactorSolverType('mumps')

    ksp_u.setType("gmres")
    ksp_u.getPC().setType('hypre')
    # ksp_u.getPC().setFactorSolverType('mumps')
    ksp_p.setType("gmres")
    ksp_p.getPC().setType('hypre')
    # ksp_p.getPC().setFactorSolverType('mumps')

opts = PETSc.Options()
opts["snes_monitor"] = None
opts["snes_converged_reason"] = None
# opts["snes_linesearch_damping"] = 0.2
# opts["snes_linesearch_type"] = "basic"
# opts["snes_linesearch_max_it"] = 1
# opts["ksp_monitor"] = None
snes.setFromOptions()

snes.setFunction(problem.F, Fvec)
snes.setJacobian(problem.J, Jmat, Pmat)

# # Initial step
# dt.value = 0.0
# for un_, u_ in zip(Un, U):
#     un_.vector.array = u_.vector.array_r
#     un_.x.scatter_forward()
#
# for n_val in [1.0, 2.0]:#, 3.0, 4.0, 5.0, 6.0]:
#     print0("Updating n value for initial solve: %f" % n_val)
#     n.value = n_val
#     snes.solve(None, x0)
#     print0("SNES and KSP convergence status: (%d, %d)" %
#          (snes.getConvergedReason(), snes.getKSP().getConvergedReason()))

# dt.value = 1.25e-2 * hmin#/10.0

# for un_, u_ in zip(Un, U):
#     un_.vector.array = u_.vector.array_r
#     un_.x.scatter_forward()
# snes.solve(None, x0)

with dolfinx.io.VTXWriter(mesh.comm, "output.bp", phi, "bp4") as fi:
    fi.write(0.0)
# for xdmf_i in range(4):
#     XDMFFile(MPI.comm_world, "out_2d_%d.xdmf" % xdmf_i).write_checkpoint(U[xdmf_i], "u", time_step=float(0))

    for j in range(n_t_steps):
        t.value = t.value + dt.value

        print0(f"Time step: {j} of {n_t_steps}, "
               f"dt={dt.value:.3e}, t={t.value:.3e}")

        for un_, u_ in zip(Un, U):
            un_.vector.array = u_.vector.array_r
            un_.x.scatter_forward()

        snes.solve(None, x0)

        fi.write(t.value)
# dolfin.list_timings([dolfin.TimingType.wall])