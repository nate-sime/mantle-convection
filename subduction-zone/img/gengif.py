import pathlib
import json

from mpi4py import MPI
import numpy as np
import adios2
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patheffects

import h5py

comm = MPI.COMM_WORLD

input_directory = pathlib.Path("../evolving2d")
output_directory = pathlib.Path("../evolving2d_results")

with open(input_directory / "metadata.json", "r") as fi:
    meta_data = json.load(fi)
t_final_yr = meta_data["t_final_yr"]
n_slab_steps = meta_data["n_slab_steps"]
idx_fmt = meta_data["idx_fmt"]
t_steps_yr = np.linspace(0.0, t_final_yr, n_slab_steps + 1)


with h5py.File(output_directory / f"temperature_{n_slab_steps:{idx_fmt}}.h5",
               "r") as fi:
    T_x = fi["Mesh"]["zone"]["geometry"][:]
    xlim = T_x[:,0].min(), T_x[:,0].max()
    ylim = T_x[:,1].min(), T_x[:,1].max()


def add_label(ax, label):
    txt = ax.text(
        0.025, 0.025, label,
        transform=ax.transAxes,
        ha="left", va="bottom",
        rotation="horizontal", weight="bold")
    txt.set_path_effects(
        [matplotlib.patheffects.withStroke(linewidth=4, foreground='w')])


def render_frame(i):
    plt.gca().clear()
    # plt.clf()
    plt.gca().set_xlim(xlim)
    plt.gca().set_ylim(ylim)
    plt.gca().set_xlabel(r"$x$ ($\mathrm{km})$")
    plt.gca().set_ylabel(r"$y$ ($\mathrm{km})$")
    plt.gcf().subplots_adjust(left=0.125, bottom=0, right=0.975, top=1,
                              wspace=None, hspace=None)
    plt.gca().set_aspect("equal")

    print(i, t_steps_yr[i])
    add_label(plt.gca(), rf"$t = {t_steps_yr[i] / 1e6:.2f} \, \mathrm{{Myr}}$")

    print("loading", i)
    with h5py.File(output_directory / f"temperature_{i:{idx_fmt}}.h5",
                   "r") as fi:
        key = list(fi["Function"]["f"].keys())
        assert len(key) == 1
        key = key[0]
        T_data = fi["Function"]["f"][key][:]
        T_x = fi["Mesh"]["zone"]["geometry"][:]
        T_cells = fi["Mesh"]["zone"]["topology"][:]

    with h5py.File(input_directory / f"subduction_zone_{i:{idx_fmt}}.h5",
                   "r") as fi:
        facets = fi["MeshTags"]["zone_facets"]["topology"][:]
        mesh_x = fi["Mesh"]["zone"]["geometry"][:]

    lc = matplotlib.collections.LineCollection(
        mesh_x[facets], colors="white", linewidths=1.5)
    plt.gca().add_collection(lc)

    tri = matplotlib.tri.Triangulation(T_x[:, 0], T_x[:, 1], triangles=T_cells)
    plt.tricontourf(tri, T_data.ravel() - 273.0, vmin=0, vmax=1400.0,
                    cmap="inferno")


import matplotlib.animation
ani = matplotlib.animation.FuncAnimation(plt.gcf(), render_frame, repeat=False,
                                         frames = n_slab_steps + 1, interval=10)
ani.save("evolving2d.gif", writer=matplotlib.animation.PillowWriter(fps=8))
