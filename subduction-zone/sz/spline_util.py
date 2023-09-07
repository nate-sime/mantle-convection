import numpy as np
import scipy
import geomdl
import geomdl.abstract
import dolfinx


def slab_velocity_kdtree(
        slab_spline: geomdl.abstract.Curve | geomdl.abstract.Surface,
        slab_spline_m: geomdl.abstract.Curve | geomdl.abstract.Surface,
        u_slab: dolfinx.fem.Function, slab_facet_indices: list[int],
        dt_val: float, resolution: int=256):
    mesh = u_slab.function_space.mesh

    # Evaluate the splines at a net of reference coordinates
    eta = np.linspace(0, 1, resolution)
    if u_slab.function_space.mesh.geometry.dim == 3:
        eta = np.vstack(map(np.ravel, np.meshgrid(eta, eta))).T
    snet_glob = np.array(slab_spline.evaluate_list(eta.tolist()))
    snet_glob_m = np.array(slab_spline_m.evaluate_list(eta.tolist()))

    # Use a kdtree to find the nearest point in the net
    d = snet_glob - snet_glob_m
    kdtree = scipy.spatial.cKDTree(snet_glob)

    # Evaluate the DoF coords of the velocity function on the provided facets
    slab_dofs = dolfinx.fem.locate_dofs_topological(
        u_slab.function_space, mesh.topology.dim - 1, slab_facet_indices)
    dof_coords = u_slab.function_space.tabulate_dof_coordinates()[
                 :,:mesh.geometry.dim]
    dof_coords_slab = dof_coords[slab_dofs]

    # Update the velocity function with the closest deformation velocity in
    # the net
    r, idx = kdtree.query(dof_coords_slab)
    bs = u_slab.function_space.dofmap.bs
    slab_dofs_unrolled = np.fromiter(
        ((dof * bs + k) for dof in slab_dofs for k in range(bs)), dtype=np.int32)
    with u_slab.vector.localForm() as lvec:
        lvec.array[slab_dofs_unrolled] += d[idx].flatten() / dt_val
    u_slab.x.scatter_forward()
