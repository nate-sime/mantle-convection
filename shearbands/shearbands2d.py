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
MatrixType = solver_utils.MatrixType


def print0(*args):
    PETSc.Sys.Print(", ".join(map(str, args)))


with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "disk.xdmf", "r") as fi:
    mesh = fi.read_mesh(
        name="mesh", ghost_mode=dolfinx.cpp.mesh.GhostMode.none)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, 0)
    facet_tags = fi.read_meshtags(mesh, name="facets")

dx = ufl.dx(metadata={"quadrature_degree": 4})
matrix_type = MatrixType.block

# Material coefficients
h_on_delta = dolfinx.fem.Constant(mesh, 4.0)
eta_0 = dolfinx.fem.Constant(mesh, 1.0)
phi_0 = dolfinx.fem.Constant(mesh, 0.05)
alpha = dolfinx.fem.Constant(mesh, -27.0)
m = dolfinx.fem.Constant(mesh, 1.0)
ghat = dolfinx.fem.Constant(mesh, [0.0] * mesh.topology.dim)
face_n = ufl.FacetNormal(mesh)

# Strainrate exponent. The target value will be approached using a continuation
# method prior to simulation. The value with which n is initialised will be
# the initial step in the linear continuation process.
n_val_target = 4.0
n = dolfinx.fem.Constant(mesh, 2.0)
nexp = (1 - n) / n

# Function spaces
p = 2
CGp = ufl.FiniteElement("CG", mesh.ufl_cell(), p)
CGk = ufl.FiniteElement("CG", mesh.ufl_cell(), p - 1)
CGp_vec = ufl.VectorElement("CG", mesh.ufl_cell(), p)
QC = dolfinx.fem.FunctionSpace(mesh, CGp)
PHI = dolfinx.fem.FunctionSpace(mesh, CGp)
Q = dolfinx.fem.FunctionSpace(mesh, CGk)
V = dolfinx.fem.FunctionSpace(mesh, CGp_vec)

u, p, phi, p_c = map(dolfinx.fem.Function, (V, Q, PHI, QC))
u_n, p_n, phi_n, p_c_n = map(dolfinx.fem.Function, (V, Q, PHI, QC))
v, q, psi, q_c = map(ufl.TestFunction, (V, Q, PHI, QC))

# Initial condition
phi.interpolate(lambda x: 1.0 + 0.01*(0.5 - np.random.rand(x.shape[1])))
phi.x.scatter_forward()

count_dofs = lambda V: V.dofmap.index_map.size_global * V.dofmap.index_map_bs
print0(f"Problem dim: (Qc: {count_dofs(QC):,} "
       f"Phi: {count_dofs(PHI):,} Q: {count_dofs(Q):,} V: {count_dofs(V):,})")

# Boundary conditions
def rot_flow(x: np.ndarray, direction: int):
    assert direction in (1, 2)
    direction_switch = (-1)**direction
    cw = [direction_switch * x[1], - direction_switch * x[0]]
    cw += [np.zeros_like(x[0])] if mesh.topology.dim == 3 else []
    cw = np.stack(cw)
    # cw_norm = np.linalg.norm(cw, axis=0)
    return cw #/ cw_norm

if mesh.topology.dim == 2:
    u_inner = dolfinx.fem.Function(V)
    u_inner.interpolate(lambda x: 0.0*rot_flow(x, 1))
    u_inner.x.scatter_forward()
    u_outer = dolfinx.fem.Function(V)
    u_outer.interpolate(lambda x: rot_flow(x, 2))
    u_outer.x.scatter_forward()

    facets_inner = facet_tags.indices[facet_tags.values == Labels.inner_face]
    bc_top = dolfinx.fem.dirichletbc(
        u_inner, dolfinx.fem.locate_dofs_topological(
            V, mesh.topology.dim - 1, facets_inner))

    facets_outer = facet_tags.indices[facet_tags.values == Labels.outer_face]
    bc_bottom = dolfinx.fem.dirichletbc(
        u_outer, dolfinx.fem.locate_dofs_topological(
            V, mesh.topology.dim - 1, facets_outer))

    bcs_V = [bc_top, bc_bottom]
elif mesh.topology.dim == 3:
    max_h = 1.0
    u_bdry = dolfinx.fem.Function(V)
    u_bdry.interpolate(lambda x: 8.0 / 3.0 * rot_flow(x, 1) * (x[2] / max_h))
    u_bdry.x.scatter_forward()

    facets_ext = dolfinx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim - 1,
        lambda x: np.ones_like(x[0], dtype=np.int8))
    u_bc = dolfinx.fem.dirichletbc(
        u_bdry, dolfinx.fem.locate_dofs_topological(
            V, mesh.topology.dim - 1, facets_ext))

    bcs_V = [u_bc]

h = dolfinx.fem.Function(dolfinx.fem.FunctionSpace(mesh, ("DG", 0)))
h.vector.array[:] = dolfinx.cpp.mesh.h(
    mesh._cpp_object, mesh.topology.dim, np.arange(
        mesh.topology.index_map(mesh.topology.dim).size_local, dtype=np.int32))

h_measure = dolfinx.cpp.mesh.h(
    mesh._cpp_object, mesh.topology.dim, np.arange(
        mesh.topology.index_map(mesh.topology.dim).size_local, dtype=np.int32))
hmin = mesh.comm.allreduce(h_measure.min(), op=MPI.MIN)
dt = dolfinx.fem.Constant(mesh, 1.25e-1 * hmin)
t = dolfinx.fem.Constant(mesh, 0.0)

# Theta scheme
u_th = 0.5*(u + u_n)
p_th = 0.5*(p + p_n)
phi_th = 0.5*(phi + phi_n)
p_c_th = 0.5*(p_c + p_c_n)

# Variational formulations
def eps(u):
    return ufl.sym(ufl.grad(u))

def eta(u, phi):
    eps_II = ufl.sqrt(0.5 * ufl.inner(eps(u), eps(u))
                      + dolfinx.fem.Constant(mesh, 1e-9))
    return eta_0 * ufl.exp(alpha*phi_0*(phi - 1.0)) * eps_II ** nexp

def sigma(u, phi):
    return 2*eta(u, phi)*eps(u)

K = phi_th**n
zeta = phi_th**(-m)
xi = eta(u_th, phi_th)*(zeta - 2.0/3.0*phi_0**m)

f1 = (
        (phi - phi_n) * psi * dx
        + dt * (
                ufl.inner(u_th, ufl.grad(phi_th))*psi*dx
                - ufl.inner(phi_0**(m-1)*(1 - phi_0*phi_th) * p_c_th / xi,  psi) * dx
        )
)
f2 = (
        ufl.inner(K * (ufl.grad(p_c) + ufl.grad(p)), ufl.grad(q_c)) * dx
        + ufl.inner(h_on_delta ** 2 * p_c / xi, q_c) * dx
)
f3 = (
        ufl.inner(sigma(u, phi), eps(v))*dx - ufl.inner(p, ufl.div(v))*dx
)
f4 = (
        -ufl.inner(ufl.div(u), q)*dx
        + ufl.inner(phi_0**m*p_c/xi, q)*dx
)

# Collect solution variables and matrices
F = [f1, f2, f3, f4]
U = [phi, p_c, u, p]
Un = [phi_n, p_c_n, u_n, p_n]
dU = list(map(ufl.TrialFunction, (PHI, QC, V, Q)))
J = solver_utils.derivative_block(F, U, dU)

dp = ufl.TrialFunction(Q)

P = [[J[0][0], J[0][1], J[0][2], J[0][3]],
     [J[1][0], J[1][1], J[1][2], J[1][3]],
     [J[2][0], J[2][1], J[2][2], J[2][3]],
     [J[3][0], J[3][1], J[3][2], -eta(u_th, phi_th)**-1*dp*q*dx]]

if matrix_type is MatrixType.block:
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

# Construct nonlinear solvers
F, J, P = map(dolfinx.fem.form, (F, J, P))
problem = solver_utils.NonlinearPDE_SNESProblem(F, J, U, bcs_V, P)
problem.F = problem.F_block if matrix_type is MatrixType.block \
    else problem.F_nest
problem.J = problem.J_block if matrix_type is MatrixType.block \
    else problem.J_nest

if matrix_type is MatrixType.block:
    Jmat = dolfinx.fem.petsc.create_matrix_block(J)
    Pmat = None
    Fvec = dolfinx.fem.petsc.create_vector_block(F)
    x0 = dolfinx.fem.petsc.create_vector_block(F)
elif matrix_type is MatrixType.nest:
    Jmat = dolfinx.fem.petsc.create_matrix_nest(J)
    Pmat = dolfinx.fem.petsc.create_matrix_nest(P)
    Fvec = dolfinx.fem.petsc.create_vector_nest(F)
    x0 = dolfinx.fem.petsc.create_vector_nest(F)

# Scatter local vectors for initial guess
if matrix_type is MatrixType.block:
    array_r = [u_.vector.array_r for u_ in U]
    im_bs = [(u_.function_space.dofmap.index_map,
              u_.function_space.dofmap.index_map_bs)
             for u_ in U]
    dolfinx.cpp.la.petsc.scatter_local_vectors(x0, array_r, im_bs)
    x0.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                   mode=PETSc.ScatterMode.FORWARD)
else:
    for u_, sub_vec in zip(U, x0.getNestSubVecs()):
        sub_vec.array[:] = u_.vector.array_r
        sub_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                            mode=PETSc.ScatterMode.FORWARD)

# Setup solvers
snes = PETSc.SNES().create(mesh.comm)
snes.setTolerances(atol=1e-7, max_it=200)

if matrix_type is MatrixType.block:
    # For the monolithic blocked problem employ direct solver, intended for 2D
    snes.getKSP().setType("preonly")
    snes.getKSP().getPC().setType("lu")
    snes.getKSP().getPC().setFactorSolverType("mumps")
else:
    # Approximate method for nested matrices, intended for 3D
    nested_IS = Jmat.getNestISs()
    snes.getKSP().setType("fgmres")
    snes.getKSP().getPC().setType("fieldsplit")
    snes.getKSP().getPC().setFieldSplitType(PETSc.PC.CompositeType.SYMMETRIC_MULTIPLICATIVE)
    snes.getKSP().getPC().setFieldSplitIS(["phi", nested_IS[0][0]],
                                          ["p_c", nested_IS[0][1]],
                                          ["u", nested_IS[0][2]],
                                          ["p", nested_IS[0][3]])

    ksp_phi, ksp_p_c, ksp_u, ksp_p = snes.getKSP().getPC().getFieldSplitSubKSP()
    ksp_phi.setType("gmres")
    ksp_phi.getPC().setType('bjacobi')
    ksp_p_c.setType("gmres")
    ksp_p_c.getPC().setType('bjacobi')
    ksp_u.setType("preonly")
    ksp_u.getPC().setType('hypre')
    ksp_p.setType("gmres")
    ksp_p.getPC().setType('bjacobi')

opts = PETSc.Options()
opts["snes_monitor"] = None
if matrix_type is MatrixType.nest:
    opts["ksp_monitor"] = None
    opts["ksp_rtol"] = 1e-6
    opts["ksp_atol"] = 1e-8
    opts["ksp_max_it"] = 50
opts["snes_converged_reason"] = None
snes.setFromOptions()

snes.setFunction(problem.F, Fvec)
snes.setJacobian(problem.J, Jmat, Pmat)

# Speed interpolation used CFL criterion estimate
u_spd = u.sub(0).collapse()
u_spd_expr = dolfinx.fem.Expression(
    ufl.sqrt(u_n**2), u_spd.function_space.element.interpolation_points())

with dolfinx.io.VTXWriter(mesh.comm, "output.bp", phi, "bp4") as fi:
    fi.write(0.0)

    # Initial continuation steps
    if not np.isclose(n.value, n_val_target):
        for un_, u_ in zip(Un, U):
            un_.vector.array = u_.vector.array_r
            un_.x.scatter_forward()
        for n_val in np.linspace(2.0, n_val_target, 5):
            PETSc.Sys.Print(f"Continuation: n_val = {n_val}")
            n.value = n_val
            snes.solve(None, x0)

    for j in range(max_steps := 1000):
        # Update previous time step variables
        for un_, u_ in zip(Un, U):
            un_.vector.array = u_.vector.array_r
            un_.x.scatter_forward()

        if j > 0:
            u_spd.interpolate(u_spd_expr)
            u_spd.x.scatter_forward()
            dt.value = (c_cfl := 2.0) * hmin / u_spd.vector.max()[1]
        t.value = t.value + dt.value
        print0(f"Time step: {j} of {max_steps}, "
               f"dt={dt.value:.3e}, t={t.value:.3e}")

        # solver_utils.NonlinearPDE_SNESProblem handles ghosts and solution
        # updates
        snes.solve(None, x0)

        if snes.getConvergedReason() < 0:
            PETSc.Sys.Print("SNES did not converge. Ending simulation.")
            break

        fi.write(t.value)
