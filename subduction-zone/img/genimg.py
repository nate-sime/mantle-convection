from mpi4py import MPI
import numpy as np
import adios2
import matplotlib.pyplot as plt
import matplotlib

import h5py

comm = MPI.COMM_WORLD

with adios2.open("../temperature.bp", "r", MPI.COMM_SELF) as fh:
	for fstep in fh:
		step_vars = fstep.available_variables()

		T = fstep.read("T")
		topo = fstep.read("connectivity")
		x = fstep.read("geometry")

with h5py.File("../subduction_zone.h5", "r") as fi:
	facets = fi["MeshTags"]["zone_facets"]["topology"][:]
	x_xdmf = fi["Mesh"]["zone"]["geometry"][:]
lc = matplotlib.collections.LineCollection(x_xdmf[facets], colors="white", linewidths=1.5)
plt.gca().add_collection(lc)

tri = matplotlib.tri.Triangulation(x[:,0], x[:,1], triangles=topo[:,1:4])
triplt = plt.tricontourf(tri, T.ravel() - 273.0, vmin=0, vmax=1400.0, cmap="inferno")
plt.xlabel(r"$x$ ($\mathrm{km})$")
plt.ylabel(r"$y$ ($\mathrm{km})$")

plt.gca().set_aspect("equal")

plt.savefig("subduction2d_iso.png", bbox_inches="tight")

plt.figure(2)
plt.colorbar(triplt, label=r"$T$ $(^\circ C)$", location="bottom", ax=plt.gca())
plt.gca().remove()
plt.savefig("colorbar.png", bbox_inches="tight")

plt.show()
