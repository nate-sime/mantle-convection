from mpi4py import MPI
import numpy as np
import adios4dolfinx
import dolfinx

import matplotlib.pyplot as plt

mesh = adios4dolfinx.read_mesh(
    MPI.COMM_WORLD, f"../checkpoint_{0:04}.bp", "BP4",
    dolfinx.mesh.GhostMode.none)
phi = dolfinx.fem.Function(dolfinx.fem.FunctionSpace(mesh, ("CG", 2)))

xlim = (-2.0, 2.0)
ylim = (-2.0, 2.0)

eps = 1e-4
nn = 64
r, t = np.mgrid[1+eps:2-eps:nn*1j,0:2*np.pi:int(2*np.pi*nn)*1j]

x, y = r * np.cos(t), r * np.sin(t)
xyz = np.c_[x.ravel(), y.ravel(), np.zeros_like(y.ravel())]

bb_tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)

cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, xyz)
colliding_cells = dolfinx.geometry.compute_colliding_cells(
    mesh, cell_candidates, xyz)
cells = colliding_cells.array[colliding_cells.offsets[:-1]]


def render_frame(i):
    print(f"Rendering frame {i}")
    plt.gca().clear()
    plt.gcf().set_size_inches(2, 2)

    adios4dolfinx.read_function(
        phi, f"../checkpoint_{i:04}.bp", "BP4")
    phi_values = phi.eval(xyz, cells).reshape(x.shape)
    plt.pcolor(x, y, phi_values, vmin=0.995, vmax=1.005, cmap="Greys")

    plt.gca().set_xlim(xlim)
    plt.gca().set_ylim(ylim)
    plt.gcf().subplots_adjust(left=0, bottom=0, right=1, top=1,
                              wspace=None, hspace=None)
    plt.gca().axis("off")
    plt.gca().set_aspect("equal")


import matplotlib.animation
nsteps = 24
ani = matplotlib.animation.FuncAnimation(plt.gcf(), render_frame, repeat=False,
                                         frames=nsteps)
ani.save("shearbands.gif", writer=matplotlib.animation.PillowWriter(fps=8))
