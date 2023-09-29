import numpy as np
import scipy
import geomdl
import geomdl.abstract
import dolfinx


def deforming_slab_velocity_kdtree(
        slab_spline: geomdl.abstract.Curve | geomdl.abstract.Surface,
        slab_spline_m: geomdl.abstract.Curve | geomdl.abstract.Surface,
        u_slab: dolfinx.fem.Function, slab_facet_indices: list[int],
        dt_val: float, resolution: int = 256) -> None:
    """
    Modify the velocity function intended for use as a boundary condition on
    the subducting slab interface to incorporate the velocity of deformation
    of that interface.

    Notes:
        The deformation velocity is found by using a kdtree (provided by SciPy)
        to find the reference space coordinate on the deforming spline.

    Args:
        slab_spline: The subduction interface spline at current time step
        slab_spline_m: The subduction interface spline at previous time step
        u_slab: The velocity function of the subduction zone slab interface to
         be used as a boundary condition
        slab_facet_indices: The facet indices associtated with the interface.
         The DoFs topologically associated with these facets will be those
          modified in `u_slab`.
        dt_val: The scaled time step
        resolution: The resolution used in the `kdtree` in each orthogonal
         spline reference axis
    """
    mesh = u_slab.function_space.mesh

    # Evaluate the splines at a net of reference coordinates
    eta = np.linspace(0, 1, resolution)
    if u_slab.function_space.mesh.geometry.dim == 3:
        eta = np.vstack(list(map(np.ravel, np.meshgrid(eta, eta)))).T
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
