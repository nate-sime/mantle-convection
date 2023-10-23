from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import pathlib

comm = MPI.COMM_WORLD

cases = ["Blankenbach_1a", "Blankenbach_2a",
         "Tosi_1", "Tosi_2", "Tosi_3", "Tosi_4"]
formulation = "C0_SIPG"
p = 3
n_ele = 128

import dolfinx
import adios4dolfinx

n = (128)*1j
XY = np.mgrid[0:1:n, 0:1:n]
X, Y = XY
x = np.c_[XY.reshape(2, -1)].T
x = np.c_[x, np.zeros_like(x[:,0])]  # Padding for 2d geometry

for case in cases:
    prefix = f"{case}_{formulation}_p{p}_n{n_ele}"
    finame = pathlib.Path(f"../checkpoints")
    mesh = adios4dolfinx.read_mesh(
        MPI.COMM_WORLD, finame / (prefix + "_velocity.bp"), "bp4",
        dolfinx.mesh.GhostMode.none)
    u = dolfinx.fem.Function(dolfinx.fem.FunctionSpace(mesh, ("DG", p - 1, (2, ))))
    adios4dolfinx.read_function(u, finame / (prefix + "_velocity.bp"), "bp4")

    T = dolfinx.fem.Function(dolfinx.fem.FunctionSpace(mesh, ("CG", p)))
    adios4dolfinx.read_function(T, finame / (prefix + "_temperature.bp"), "bp4")

    mu = dolfinx.fem.Function(dolfinx.fem.FunctionSpace(mesh, ("DG", p - 2)))
    adios4dolfinx.read_function(mu, finame / (prefix + "_viscosity.bp"), "bp4")

    # Find cells
    bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, x)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(
        mesh, cell_candidates, x)
    cells = colliding_cells.array[colliding_cells.offsets[:-1]]

    # Temperature plot
    plt.figure(1, figsize=(2, 2))
    T_data = T.eval(x, cells)
    triplt = plt.tricontourf(x[:,0], x[:,1], T_data.ravel(), vmin=0, vmax=1.0,
                             cmap="inferno")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.savefig(f"{case}_temperature.png", bbox_inches="tight", pad_inches=0)
    plt.clf()

    # Velocity plot
    u_data = u.eval(x, cells)
    U, V = u_data[:,0], u_data[:,1]
    speed = np.sqrt(U**2 + V**2)
    tcf = plt.tricontourf(x[:,0], x[:,1], speed / speed.max(), cmap="Blues")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.savefig(f"{case}_velocity.png", bbox_inches="tight", pad_inches=0)
    plt.clf()

    # Viscosity plot
    mu_data = mu.eval(x, cells)
    lev_exp = np.arange(-5.0, 2.0)
    levs = np.power(10, lev_exp)

    import matplotlib.colors
    import matplotlib.ticker
    triplt = plt.tricontourf(x[:,0], x[:,1], mu_data.ravel()-1e-10, levs,
                             locator=matplotlib.ticker.LogLocator(),
                             norm=matplotlib.colors.LogNorm(), cmap="viridis")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.savefig(f"{case}_viscosity.png", bbox_inches="tight", pad_inches=0)
    plt.clf()
