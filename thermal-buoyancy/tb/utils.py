from petsc4py import PETSc
from mpi4py import MPI
import dolfinx


def create_snes(problem, F, J, max_it, rtol=1e-14, atol=1e-14, extra_opts={}):
    snes = PETSc.SNES().create(MPI.COMM_WORLD)
    snes.max_it = max_it
    snes.setTolerances(rtol=rtol, atol=atol)
    opts = PETSc.Options()
    # opts["snes_monitor"] = None
    for k, v in extra_opts.items():
        opts[k] = v
    snes.setFromOptions()
    snes.getKSP().getPC().setType("lu")
    snes.getKSP().getPC().setFactorSolverType("mumps")
    snes.setFunction(problem.F_mono, dolfinx.fem.petsc.create_vector(F))
    snes.setJacobian(problem.J_mono, J=dolfinx.fem.petsc.create_matrix(J))
    snes.setTolerances(rtol=1e-14, atol=1e-14)
    return snes


def create_bc_geo_marker(V, marker, value):
    if isinstance(V, (tuple, list)):
        mesh = V[0].mesh
    else:
        mesh = V.mesh
    facets = dolfinx.mesh.locate_entities_boundary(
        mesh, dim=mesh.topology.dim-1, marker=marker)
    dofs = dolfinx.fem.locate_dofs_topological(
        V, mesh.topology.dim - 1, facets)
    if isinstance(V, (tuple, list)):
        bc = dolfinx.fem.dirichletbc(value, dofs, V[0])
    else:
        bc = dolfinx.fem.dirichletbc(value, dofs, V)
    return bc


class NonlinearPDE_SNESProblem():
    def __init__(self, F, J, soln_vars, bcs, P=None):
        self.L = F
        self.a = J
        self.a_precon = P
        self.bcs = bcs
        self.soln_vars = soln_vars

    def F_mono(self, snes, x, F):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        with x.localForm() as _x:
            self.soln_vars.x.array[:] = _x.array_r
        with F.localForm() as f_local:
            f_local.set(0.0)
        dolfinx.fem.petsc.assemble_vector(F, self.L)
        dolfinx.fem.apply_lifting(
            F, [self.a], bcs=[self.bcs], x0=[x], scale=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.set_bc(F, self.bcs, x, -1.0)

    def J_mono(self, snes, x, J, P):
        J.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(J, self.a, bcs=self.bcs, diagonal=1.0)
        J.assemble()
        if self.a_precon is not None:
            P.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix(
                P, self.a_precon, bcs=self.bcs, diagonal=1.0)
            P.assemble()

    def F_block(self, snes, x, F):
        assert x.getType() != "nest"
        assert F.getType() != "nest"
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                      mode=PETSc.ScatterMode.FORWARD)
        with F.localForm() as f_local:
            f_local.set(0.0)

        offset = 0
        x_array = x.getArray(readonly=True)
        for var in self.soln_vars:
            size_local = var.vector.getLocalSize()
            var.vector.array[:] = x_array[offset: offset + size_local]
            var.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                   mode=PETSc.ScatterMode.FORWARD)
            offset += size_local

        dolfinx.fem.petsc.assemble_vector_block(
            F, self.L, self.a, bcs=self.bcs, x0=x, scale=-1.0)

    def J_block(self, snes, x, J, P):
        assert x.getType() != "nest" and J.getType() != "nest" \
               and P.getType() != "nest"
        J.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix_block(
            J, self.a, bcs=self.bcs, diagonal=1.0)
        J.assemble()
        if self.a_precon is not None:
            P.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix_block(
                P, self.a_precon, bcs=self.bcs, diagonal=1.0)
            P.assemble()

    def F_nest(self, snes, x, F):
        assert x.getType() == "nest" and F.getType() == "nest"
        # Update solution
        x = x.getNestSubVecs()
        for x_sub, var_sub in zip(x, self.soln_vars):
            x_sub.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                              mode=PETSc.ScatterMode.FORWARD)
            with x_sub.localForm() as _x:
                var_sub.x.array[:] = _x.array_r

        # Assemble
        bcs1 = dolfinx.fem.bcs_by_block(
            dolfinx.fem.extract_function_spaces(self.a, 1), self.bcs)
        for L, F_sub, a in zip(self.L, F.getNestSubVecs(), self.a):
            with F_sub.localForm() as F_sub_local:
                F_sub_local.set(0.0)
            dolfinx.fem.petsc.assemble_vector(F_sub, L)
            dolfinx.fem.apply_lifting(F_sub, a, bcs=bcs1, x0=x, scale=-1.0)
            F_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # Set bc value in RHS
        bcs0 = dolfinx.fem.bcs_by_block(
            dolfinx.fem.extract_function_spaces(self.L), self.bcs)
        for F_sub, bc, x_sub in zip(F.getNestSubVecs(), bcs0, x):
            dolfinx.fem.set_bc(F_sub, bc, x_sub, -1.0)

        # Must assemble F here in the case of nest matrices
        F.assemble()

    def J_nest(self, snes, x, J, P):
        assert J.getType() == "nest" and P.getType() == "nest"
        J.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix_nest(J, self.a, bcs=self.bcs, diagonal=1.0)
        J.assemble()
        if self.a_precon is not None:
            P.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix_nest(
                P, self.a_precon, bcs=self.bcs, diagonal=1.0)
            P.assemble()

