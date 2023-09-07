import numpy as np
import dolfinx

def transfer_facet_tags(mesh, facet_tags, entity_map, submesh):
    """
    Transfer the facets_tags from mesh to submesh

    Notes:
        All values in the facet_tags are assumed positive

    Args:
        mesh: Original mesh
        facet_tags: Facet tags defined on original mesh
        entity_map: Relationship from submesh child entity to parent mesh
         entity
        submesh: The submesh of the original mesh

    Returns:
        Facet tags transferred to the submesh
    """
    tdim = mesh.topology.dim
    fdim = mesh.topology.dim - 1

    f_map = mesh.topology.index_map(fdim)
    all_facets = f_map.size_local + f_map.num_ghosts
    all_values = np.full(all_facets, -1, dtype=np.int32)
    all_values[facet_tags.indices] = facet_tags.values
    c_to_f = mesh.topology.connectivity(tdim, fdim)

    submesh.topology.create_entities(fdim)
    subf_map = submesh.topology.index_map(fdim)
    submesh.topology.create_connectivity(tdim, fdim)
    c_to_f_sub = submesh.topology.connectivity(tdim, fdim)
    num_sub_facets = subf_map.size_local + subf_map.num_ghosts
    sub_values = np.empty(num_sub_facets, dtype=np.int32)
    for i, entity in enumerate(entity_map):
        parent_facets = c_to_f.links(entity)
        child_facets = c_to_f_sub.links(i)
        for child, parent in zip(child_facets, parent_facets):
            sub_values[child] = all_values[parent]

    valid_entries = ~(sub_values == -1)
    sub_indices = np.arange(num_sub_facets, dtype=np.int32)[valid_entries]
    sub_values = sub_values[valid_entries]

    sub_meshtag = dolfinx.mesh.meshtags(
        submesh, submesh.topology.dim-1,
        sub_indices, sub_values)
    return sub_meshtag


def extract_submesh_and_transfer_facets(mesh, facet_tags, indices):
    sub_mesh, entity_map, _, _ = dolfinx.mesh.create_submesh(
        mesh, mesh.topology.dim, indices)
    sub_mesh.topology.create_connectivity(
        sub_mesh.topology.dim - 1, sub_mesh.topology.dim)
    sub_facet_tags = transfer_facet_tags(
        mesh, facet_tags, entity_map, sub_mesh)
    return sub_mesh, sub_facet_tags
