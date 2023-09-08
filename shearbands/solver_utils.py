import enum

from petsc4py import PETSc
import dolfinx
import ufl


def derivative_block(F, u, du=None, coefficient_derivatives=None):
    """
    Parameters
    ----------
    F
        Block residual formulation
    u
        Ordered solution functions
    du
        Ordered trial functions
    coefficient_derivatives
        Prescribed derivative map

    Returns
    -------
    Block matrix corresponding to the ordered components of the
    Gateaux/Frechet derivative.
    """
    if isinstance(F, ufl.Form):
        return ufl.derivative(F, u, du, coefficient_derivatives)

    if not isinstance(F, (list, tuple)):
        raise TypeError("Expecting F to be a list of Forms. Found: %s"
                        % str(F))

    if not isinstance(u, (list, tuple)):
        raise TypeError("Expecting u to be a list of Coefficients. Found: %s"
                        % str(u))

    if du is not None:
        if not isinstance(du, (list, tuple)):
            raise TypeError("Expecting du to be a list of Arguments. Found: %s"
                            % str(u))

    import itertools
    from ufl.algorithms.apply_derivatives import apply_derivatives
    from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering

    m, n = len(u), len(F)

    if du is None:
        du = [None] * m

    J = [[None for _ in range(m)] for _ in range(n)]

    for (i, j) in itertools.product(range(n), range(m)):
        gateaux_derivative = ufl.derivative(F[i], u[j], du[j],
                                            coefficient_derivatives)
        gateaux_derivative = apply_derivatives(
            apply_algebra_lowering(gateaux_derivative))
        if gateaux_derivative.empty():
            gateaux_derivative = None
        J[i][j] = gateaux_derivative

    return J


class MatrixType(enum.Enum):
    monolithic = enum.auto()
    block = enum.auto()
    nest = enum.auto()

    def is_block_type(self):
        return self is not MatrixType.monolithic


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