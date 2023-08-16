import enum
import dolfinx
from mpi4py import MPI
import gmsh
import numpy as np


class Labels(enum.IntEnum):
    plate = 1
    slab = 2
    wedge = 3
    plate_wedge = 4
    slab_plate = 5
    slab_wedge = 6
    slab_left = 7
    slab_bottom = 8
    wedge_right = 9
    wedge_bottom = 10
    plate_top = 11
    plate_right = 12


def transfer_facet_tags(mesh, facet_tags, entity_map, submesh):
    tdim = mesh.topology.dim
    fdim = mesh.topology.dim - 1

    f_map = mesh.topology.index_map(fdim)
    all_facets = f_map.size_local + f_map.num_ghosts
    all_values = np.zeros(all_facets, dtype=np.int32)
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

    sub_meshtag = dolfinx.mesh.meshtags(
        submesh, submesh.topology.dim-1,
        np.arange(num_sub_facets, dtype=np.int32), sub_values)
    return sub_meshtag

def generate(comm: MPI.Intracomm):
    gmsh.initialize()

    slab_depth = 50.0
    depth = 600.0
    dx = 60.0

    if comm.rank == 0:
        gmsh.model.add("subduction")

        # -- Geometry definition
        domain = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, depth + dx, -depth)
        tl = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)
        br = gmsh.model.occ.addPoint(depth, -depth, 0.0)
        interface = gmsh.model.occ.addLine(tl, br)

        gmsh.model.occ.fragment([(2, domain)], [(1, interface)],
                                removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()

        l = gmsh.model.occ.addPoint(slab_depth, -slab_depth, 0.0)
        r = gmsh.model.occ.addPoint(depth + dx, -slab_depth, 0.0)
        plate_iface = gmsh.model.occ.addLine(l, r)
        gmsh.model.occ.fragment(gmsh.model.occ.getEntities(2), [(1, plate_iface)],
                                removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()

        # -- Labelling
        # Volume labels
        vols = gmsh.model.occ.getEntities(2)
        coms = np.array([gmsh.model.occ.getCenterOfMass(*v) for v in vols])
        plate = vols[np.argmax(coms[:,1])]
        slab = vols[np.argmin(coms[:,0])]
        wedge = vols[np.argmax(coms[:,0])]

        gmsh.model.addPhysicalGroup(2, [plate[1]], tag=Labels.plate)
        gmsh.model.addPhysicalGroup(2, [slab[1]], tag=Labels.slab)
        gmsh.model.addPhysicalGroup(2, [wedge[1]], tag=Labels.wedge)

        # Domain interface facet labels
        plate_facets = gmsh.model.getAdjacencies(plate[0], plate[1])[1]
        slab_facets = gmsh.model.getAdjacencies(slab[0], slab[1])[1]
        wedge_facets = gmsh.model.getAdjacencies(wedge[0], wedge[1])[1]

        slab_plate = np.intersect1d(slab_facets, plate_facets)
        slab_wedge = np.intersect1d(slab_facets, wedge_facets)
        plate_wedge = np.intersect1d(plate_facets, wedge_facets)
        gmsh.model.addPhysicalGroup(1, slab_plate, tag=Labels.slab_plate)
        gmsh.model.addPhysicalGroup(1, slab_wedge, tag=Labels.slab_wedge)
        gmsh.model.addPhysicalGroup(1, plate_wedge, tag=Labels.plate_wedge)

        # Domain exterior facet labels
        def facet_coms(facets):
            return np.array([gmsh.model.occ.getCenterOfMass(1, f) for f in facets])
        coms_plate = facet_coms(plate_facets)
        coms_wedge = facet_coms(wedge_facets)
        coms_slab = facet_coms(slab_facets)

        # slab
        gmsh.model.addPhysicalGroup(1, [slab_facets[np.argmin(coms_slab[:,0])]],
                                    tag=Labels.slab_left)
        gmsh.model.addPhysicalGroup(1, [slab_facets[np.argmin(coms_slab[:,1])]],
                                    tag=Labels.slab_bottom)
        # wedge
        gmsh.model.addPhysicalGroup(1, [wedge_facets[np.argmax(coms_wedge[:,0])]],
                                    tag=Labels.wedge_right)
        gmsh.model.addPhysicalGroup(1, [wedge_facets[np.argmin(coms_wedge[:,1])]],
                                    tag=Labels.wedge_bottom)
        # plate
        gmsh.model.addPhysicalGroup(1, [plate_facets[np.argmax(coms_plate[:,0])]],
                                    tag=Labels.plate_right)
        gmsh.model.addPhysicalGroup(1, [plate_facets[np.argmax(coms_plate[:,1])]],
                                    tag=Labels.plate_top)

        # -- Local refinement
        def get_pts(dimtag):
            return gmsh.model.getBoundary([dimtag], recursive=True)
        corner_pt = set.intersection(
            *map(lambda x: set(get_pts(x)), (plate, slab, wedge)))
        assert len(corner_pt) == 1
        corner_pt = corner_pt.pop()

        corner_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(corner_dist, "PointsList", [corner_pt[1]])

        L = depth
        D = depth
        xxx = 5.0
        corner_res_min_max = (xxx/2.0, xxx)
        interface_res_min_max = (xxx/2.0, xxx)

        corner_thres = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(corner_thres, "IField", corner_dist)
        gmsh.model.mesh.field.setNumber(
            corner_thres, "LcMin", corner_res_min_max[0])
        gmsh.model.mesh.field.setNumber(
            corner_thres, "LcMax", corner_res_min_max[1])
        gmsh.model.mesh.field.setNumber(corner_thres, "DistMin", 0.1*D)
        gmsh.model.mesh.field.setNumber(corner_thres, "DistMax", 0.2*D)

        # Other surfaces to refine
        interface_dist = gmsh.model.mesh.field.add("Distance")
        refine_surfs = np.concatenate(
            ([slab_wedge[0]], gmsh.model.getEntitiesForPhysicalGroup(1, Labels.plate_wedge)))
        gmsh.model.mesh.field.setNumbers(interface_dist, "CurvesList", refine_surfs)
        interface_thresh = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(interface_thresh, "IField", interface_dist)
        gmsh.model.mesh.field.setNumber(
            interface_thresh, "LcMin", interface_res_min_max[0])
        gmsh.model.mesh.field.setNumber(
            interface_thresh, "LcMax", interface_res_min_max[1])
        gmsh.model.mesh.field.setNumber(interface_thresh, "DistMin", 0.01*D)
        gmsh.model.mesh.field.setNumber(interface_thresh, "DistMax", 0.5*D)

        field_min_all = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(
            field_min_all, "FieldsList", [interface_thresh, corner_thres])
        gmsh.model.mesh.field.setAsBackgroundMesh(field_min_all)

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.set_order(1)

    partitioner = dolfinx.mesh.create_cell_partitioner(
        dolfinx.mesh.GhostMode.none)
    mesh, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, rank=0, gdim=2, partitioner=partitioner)

    # import febug
    # febug.plot_mesh(mesh).show()
    return mesh, cell_tags, facet_tags


if __name__ == "__main__":
    from mpi4py import MPI
    generate(MPI.COMM_WORLD)