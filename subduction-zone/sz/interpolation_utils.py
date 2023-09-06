import numpy as np
from petsc4py import PETSc
import dolfinx


def nonmatching_interpolate(u1: dolfinx.fem.Function, u2: dolfinx.fem.Function,
                            padding: float=1e-12):
    """
    Given two functions which are defined on a coarse and a fine mesh,
    interpolate from the coarse representation to the fine representation
    using linear interpolation.

    Args:
        u1: Coarse function
        u2: Fine function
        padding: Buffer for numerical precision of bounding box limits

    Notes:
        This function is not as efficient as DOLFINx in-built non-matching
        mesh interpolation; however, it is more flexible for interpolating
        data not contained within the computational domain.
    """
    V2 = u2.function_space
    mesh1 = u1.function_space.mesh
    comm = mesh1.comm
    rank = comm.rank
    comm_size = comm.size

    dof_coords = V2.tabulate_dof_coordinates()
    bs = V2.dofmap.bs
    im2 = V2.dofmap.index_map

    # Find the collisions on the coarse mesh 1
    bbtree = dolfinx.geometry.bb_tree(
        mesh1, mesh1.topology.dim, padding=padding)
    bbtree_global = bbtree.create_global_tree(mesh1.comm)

    u2_n_ldofs = u2.vector.getLocalSize()
    u2_coeffs = np.zeros(u2_n_ldofs, dtype=np.double)
    off_proc_points = []
    exterior_points = []

    # -- Compute on process dofs and tabulate off process and exterior coords
    for dof, p in enumerate(dof_coords[:im2.size_local]):
        cell_candidates = dolfinx.geometry.compute_collisions_points(
            bbtree, p)
        cell = dolfinx.geometry.compute_colliding_cells(
            mesh1, cell_candidates, p)

        if len(cell) > 0:
            cell = cell[0]
            # Point is inside the domain and on this process, interpolate and
            # add to vector of coefficients
            u1_val = u1.eval(p, cell)
            u2_coeffs[dof * bs:dof * bs + bs] = u1_val
        else:
            proc_col = dolfinx.geometry.compute_collisions_points(
                bbtree_global, p)
            proc_col = proc_col.links(0)
            proc_col = [i for i in proc_col if not i == rank]
            if len(proc_col) > 0:
                # Point is on another process
                off_proc_points.append((rank, proc_col, p, dof))
            else:
                # Point is outside of the domain
                exterior_points.append((rank, p, dof))

    # -- Compute off process interpolants at dofs
    # Pack up the points on to be sent to other processes for evaluation
    data_to_send = [[] for _ in range(comm.size)]
    for source, candidates, pt, dof in off_proc_points:
        for candidate in candidates:
            data_to_send[candidate].append(
                {"point": pt, "source": source, "dof": dof})

    data_recv = comm.alltoall(data_to_send)

    # Interpolate at the points from other processes, then pack them up and
    # send those interpolants back
    data_to_send_back = [[] for _ in range(comm.size)]
    for source, data in enumerate(data_recv):
        for datum in data:
            p = datum["point"]
            cell_candidates = dolfinx.geometry.compute_collisions_points(
                bbtree, p)
            cell = dolfinx.geometry.compute_colliding_cells(
                mesh1, cell_candidates, p)

            if len(cell) > 0:
                cell = cell[0]
                u1_val = u1.eval(p, cell)
                data_to_send_back[source].append({
                    "u1_val": u1_val, "dof": datum["dof"]})

    found_off_proc_dofs = []
    received_vals = comm.alltoall(data_to_send_back)
    for proc_data in received_vals:
        for datum in proc_data:
            dof = datum["dof"]
            found_off_proc_dofs.append(dof)
            u1_val = datum["u1_val"]
            u2_coeffs[dof * bs:dof * bs + bs] = u1_val

    # Have some off process dofs been lost? If yes consider them exterior to
    # the domain. Perhaps the processing bboxes aren't quite containers
    off_proc_dofs = [datum[3] for datum in off_proc_points]
    dofs_off_proc_lost = list(set(off_proc_dofs) - set(found_off_proc_dofs))
    for source, candidates, pt, dof in off_proc_points:
        if dof in dofs_off_proc_lost:
            exterior_points.append((source, pt, dof))

    # -- Extrapolate exterior points from nearest cells
    data_to_send = [[] for _ in range(comm.size)]
    for rank, p, dof in exterior_points:
        for j in range(comm_size):
            data_to_send[j].append({"point": p, "source": source, "dof": dof})
    data_recv = comm.alltoall(data_to_send)

    # Interpolate at nearest point
    data_to_send_back = [[] for _ in range(comm.size)]
    if any(map(len, data_recv)):
        num_entities_local = mesh1.topology.index_map(
            mesh1.topology.dim).size_local + mesh1.topology.index_map(
            mesh1.topology.dim).num_ghosts
        entities = np.arange(num_entities_local, dtype=np.int32)
        mesh1_cell_midpoint_tree = dolfinx.geometry.create_midpoint_tree(
            mesh1, mesh1.topology.dim, entities)

        c2vg = dolfinx.cpp.mesh.entities_to_geometry(
            mesh1._cpp_object, mesh1.topology.dim,
            np.arange(mesh1.topology.index_map(mesh1.topology.dim).size_local,
                      dtype=np.int32),
            False)

        for source, data in enumerate(data_recv):
            for datum in data:
                p = datum["point"]
                cell = dolfinx.geometry.compute_closest_entity(
                    bbtree, mesh1_cell_midpoint_tree, mesh1, p)
                r = dolfinx.geometry.squared_distance(
                    mesh1, mesh1.topology.dim, cell, p)

                c0 = mesh1.geometry.x[c2vg[cell[0]]]
                s0 = dolfinx.geometry.compute_distance_gjk(c0, p)
                u1_val = u1.eval(p + s0, cell)
                data_to_send_back[source].append({
                    "u1_val": u1_val, "dof": datum["dof"], "r": r})
    data_recv = comm.alltoall(data_to_send_back)

    # Choose the shortest distance value to add in
    exterior_data = {}
    for proc_data in data_recv:
        for datum in proc_data:
            exterior_data.setdefault(datum["dof"], []).append(
                (datum["r"], datum["u1_val"]))

    for dof, r_v in exterior_data.items():
        r = np.fromiter((d[0] for d in r_v), dtype=np.double)
        v = np.array(list(d[1] for d in r_v), dtype=np.double)
        idx_min = np.argmin(r)
        u2_coeffs[dof * bs:dof * bs + bs] = v[idx_min]

    u2.vector.setArray(u2_coeffs)
    u2.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                          mode=PETSc.ScatterMode.FORWARD)
