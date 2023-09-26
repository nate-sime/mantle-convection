import typing
import numpy as np
import dolfinx


def transfer_meshtags(mesh: dolfinx.mesh.Mesh,
                      meshtags: dolfinx.mesh.MeshTags,
                      entity_map: dolfinx.graph.adjacencylist,
                      submesh: dolfinx.mesh.Mesh):
    """
    Transfer the meshtags from mesh to submesh

    Notes:
        All values in the meshtags are assumed positive

    Args:
        mesh: Original mesh
        meshtags: Mesh tags defined on original mesh
        entity_map: Relationship from submesh child entity to parent mesh
         entity
        submesh: The submesh of the original mesh

    Returns:
        Mesh tags transferred to the submesh
    """
    tdim = mesh.topology.dim
    mtdim = meshtags.dim

    # Use the suffix _e to correspond to the entity of the meshtag topology
    mt_map = mesh.topology.index_map(mtdim)
    all_facets = mt_map.size_local + mt_map.num_ghosts
    all_values = np.full(all_facets, -1, dtype=np.int32)
    all_values[meshtags.indices] = meshtags.values
    c_to_e = mesh.topology.connectivity(tdim, mtdim)

    submesh.topology.create_entities(mtdim)
    sub_mt_map = submesh.topology.index_map(mtdim)
    submesh.topology.create_connectivity(tdim, mtdim)
    c_to_e_sub = submesh.topology.connectivity(tdim, mtdim)
    num_sub_es = sub_mt_map.size_local + sub_mt_map.num_ghosts
    sub_values = np.empty(num_sub_es, dtype=np.int32)
    for i, entity in enumerate(entity_map):
        parent_es = c_to_e.links(entity)
        child_es = c_to_e_sub.links(i)
        for child, parent in zip(child_es, parent_es):
            sub_values[child] = all_values[parent]

    valid_entries = ~(sub_values == -1)
    sub_indices = np.arange(num_sub_es, dtype=np.int32)[valid_entries]
    sub_values = sub_values[valid_entries]

    sub_meshtag = dolfinx.mesh.meshtags(
        submesh, mtdim, sub_indices, sub_values)
    return sub_meshtag


def extract_submesh_and_transfer_meshtags(
        mesh: dolfinx.mesh.Mesh,
        meshtags: dolfinx.mesh.MeshTags | typing.Sequence[dolfinx.mesh.MeshTags],
        indices: np.ndarray):
    """
    Given an array of cell indices, extract the submesh and transfer meshtags
    belonging to the parent mesh to the submesh.

    Args:
        mesh: Parent mesh
        meshtags: Instance or sequence of meshtags
        indices: Array of cell indices to extract into a submesh

    Returns:
    The submesh and transferred meshtags with order preserved
    """
    sub_mesh, entity_map, _, _ = dolfinx.mesh.create_submesh(
        mesh, mesh.topology.dim, indices)

    if not isinstance(meshtags, (tuple, list)):
        meshtags = (meshtags,)
    meshtags_unique_dims = set(mt.dim for mt in meshtags)
    for mt_dim in meshtags_unique_dims:
        sub_mesh.topology.create_connectivity(mt_dim, sub_mesh.topology.dim)
    sub_meshtags = [transfer_meshtags(
            mesh, mt, entity_map, sub_mesh) for mt in meshtags]

    if len(sub_meshtags) == 1:
        sub_meshtags = sub_meshtags[0]
    return sub_mesh, sub_meshtags
