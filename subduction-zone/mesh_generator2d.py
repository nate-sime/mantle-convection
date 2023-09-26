import enum
import typing

import geomdl
import geomdl.abstract
import dolfinx
from mpi4py import MPI
import gmsh
import numpy as np

from sz.model import Labels
from sz.mesh_utils import extract_submesh_and_transfer_meshtags


def generate(comm: MPI.Intracomm,
             slab_spline: geomdl.abstract.Curve,
             wedge_x_buffer: float,
             plate_y: float,
             corner_resolution: float,
             surface_resolution: float,
             bulk_resolution: float,
             couple_y: float = None,
             geom_degree: int = 1):
    gmsh.initialize()
    if comm.rank == 0:
        slab_x0 = slab_spline.evaluate_single(0.0)
        slab_br = slab_spline.evaluate_single(1.0)
        slab_width = slab_br[0] - slab_x0[0]
        slab_depth = slab_br[1] - slab_x0[1]

        gmsh.model.add("subduction")

        # -- Geometry definition
        domain = gmsh.model.occ.addRectangle(
            slab_x0[0], slab_x0[1], 0.0,
            slab_width + wedge_x_buffer, slab_depth)

        cpts = [gmsh.model.occ.addPoint(*pt, 0.0) for pt in slab_spline.ctrlpts]
        interface = gmsh.model.occ.addBSpline(cpts, degree=slab_spline.degree)

        gmsh.model.occ.fragment([(2, domain)], [(1, interface)],
                                removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()

        # Generate delimiting horizontal lines of constant depth. Particularly
        # useful for depth dependent material coefficients
        def fragment_wedge_with_horizontal_line(lines):
            if not hasattr(lines, "__len__"):
                lines = (lines,)
            vols = gmsh.model.occ.getEntities(2)
            coms = np.array([gmsh.model.occ.getCenterOfMass(*v) for v in vols])
            wedge = vols[np.argmax(coms[:,1])]
            gmsh.model.occ.fragment([wedge], [(1, li) for li in lines],
                                    removeObject=True, removeTool=True)
            gmsh.model.occ.healShapes()
            gmsh.model.occ.synchronize()

        plate_iface = gmsh.model.occ.addLine(
            gmsh.model.occ.addPoint(slab_x0[0], plate_y, 0.0),
            gmsh.model.occ.addPoint(slab_br[0] + wedge_x_buffer, plate_y, 0.0))
        lines_to_fragment = [plate_iface]

        if couple_y is not None:
            assert couple_y < plate_y
            couple_line = gmsh.model.occ.addLine(
                gmsh.model.occ.addPoint(slab_x0[0], couple_y, 0.0),
                gmsh.model.occ.addPoint(slab_br[0] + wedge_x_buffer, couple_y, 0.0))
            lines_to_fragment.append(couple_line)

        fragment_wedge_with_horizontal_line(lines_to_fragment)

        # -- Labelling
        # Volume labels
        vols = gmsh.model.occ.getEntities(2)
        coms = np.array([gmsh.model.occ.getCenterOfMass(*v) for v in vols])
        plate = vols[np.argmax(coms[:,1])]
        slab = vols[np.argmin(coms[:,0])]
        wedge = list(set(vols) - set([plate, slab]))

        gmsh.model.addPhysicalGroup(2, [plate[1]], tag=Labels.plate)
        gmsh.model.addPhysicalGroup(2, [slab[1]], tag=Labels.slab)
        gmsh.model.addPhysicalGroup(2, [w[1] for w in wedge], tag=Labels.wedge)

        # Domain interface facet labels
        plate_facets = gmsh.model.getAdjacencies(plate[0], plate[1])[1]
        slab_facets = gmsh.model.getAdjacencies(slab[0], slab[1])[1]
        wedge_facets = np.fromiter(set(sum((
            gmsh.model.getAdjacencies(2, w[1])[1].tolist() for w in wedge), [])
        ), dtype=np.int32)

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
        gmsh.model.addPhysicalGroup(1,
                                    [slab_facets[np.argmin(coms_slab[:, 0])]],
                                    tag=Labels.slab_left)
        gmsh.model.addPhysicalGroup(1,
                                    [slab_facets[np.argmin(coms_slab[:, 1])]],
                                    tag=Labels.slab_bottom)
        # wedge
        gmsh.model.addPhysicalGroup(1, wedge_facets[
            np.isclose(coms_wedge[:, 0], slab_br[0] + wedge_x_buffer)],
                                    tag=Labels.wedge_right)
        gmsh.model.addPhysicalGroup(1,
                                    [wedge_facets[np.argmin(coms_wedge[:, 1])]],
                                    tag=Labels.wedge_bottom)
        # plate
        gmsh.model.addPhysicalGroup(1,
                                    [plate_facets[np.argmax(coms_plate[:, 0])]],
                                    tag=Labels.plate_right)
        gmsh.model.addPhysicalGroup(1,
                                    [plate_facets[np.argmax(coms_plate[:, 1])]],
                                    tag=Labels.plate_top)

        # -- Local refinement
        def get_pts(dimtag):
            if isinstance(dimtag, tuple):
                dimtag = [dimtag]
            return gmsh.model.getBoundary(dimtag, recursive=True)

        # Apply local refinement to the coupling point if it exists, otherwise
        # the wedge corner
        if couple_y is None:
            corner_pt = set.intersection(
                *map(lambda x: set(get_pts(x)), (plate, slab, wedge)))
            assert len(corner_pt) == 1
            corner_pt = corner_pt.pop()
        else:
            corner_pt = list(set.intersection(
                *map(lambda x: set(get_pts(x)), (slab, *(w for w in wedge)))))
            corner_coords = np.array(
                [gmsh.model.getValue(0, pt[1], []) for pt in corner_pt],
                dtype=np.float64)
            corner_pt = corner_pt[np.argmin(np.abs(corner_coords[:,1] - couple_y))]

        corner_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(corner_dist, "PointsList", [corner_pt[1]])

        L = abs(slab_width)
        D = abs(slab_depth)
        corner_res_min_max = (corner_resolution, bulk_resolution)
        interface_res_min_max = (surface_resolution, bulk_resolution)

        corner_thres = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(corner_thres, "IField", corner_dist)
        gmsh.model.mesh.field.setNumber(
            corner_thres, "LcMin", corner_res_min_max[0])
        gmsh.model.mesh.field.setNumber(
            corner_thres, "LcMax", corner_res_min_max[1])
        gmsh.model.mesh.field.setNumber(corner_thres, "DistMin", 0.05*D)
        gmsh.model.mesh.field.setNumber(corner_thres, "DistMax", 0.1*D)

        # Other surfaces to refine
        interface_dist = gmsh.model.mesh.field.add("Distance")
        refine_surfs = np.concatenate(
            (gmsh.model.getEntitiesForPhysicalGroup(1, Labels.slab_wedge),
             gmsh.model.getEntitiesForPhysicalGroup(1, Labels.plate_wedge)))
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
        gmsh.model.mesh.set_order(geom_degree)

    partitioner = dolfinx.mesh.create_cell_partitioner(
        dolfinx.mesh.GhostMode.none)
    mesh, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, rank=0, gdim=2, partitioner=partitioner)

    gmsh.finalize()
    return mesh, cell_tags, facet_tags


if __name__ == "__main__":
    from mpi4py import MPI

    depth = 600.0
    x_slab = np.linspace(0, depth, 10)
    y_slab = -x_slab
    wedge_x_buffer = 50.0
    plate_y = -50.0
    couple_y = None
    bulk_resolution = 25.0
    corner_resolution = 2.0
    surface_resolution = 5.0

    import geomdl.fitting
    slab_spline = geomdl.fitting.interpolate_curve(
        np.stack((x_slab, y_slab)).T.tolist(), degree=3)

    mesh, cell_tags, facet_tags = generate(
        MPI.COMM_WORLD, slab_spline, wedge_x_buffer, plate_y,
        corner_resolution, surface_resolution, bulk_resolution,
        couple_y=couple_y)
    cell_tags.name = "zone_cells"
    facet_tags.name = "zone_facets"
    mesh.name = "zone"

    wedge_mesh, wedge_facet_tags = extract_submesh_and_transfer_meshtags(
        mesh, facet_tags, cell_tags.indices[cell_tags.values == Labels.wedge])
    wedge_facet_tags.name = "wedge_facets"
    wedge_mesh.name = "wedge"
    slab_mesh, slab_facet_tags = extract_submesh_and_transfer_meshtags(
        mesh, facet_tags, cell_tags.indices[cell_tags.values == Labels.slab])
    slab_facet_tags.name = "slab_facets"
    slab_mesh.name = "slab"

    with dolfinx.io.XDMFFile(mesh.comm, "subduction_zone.xdmf", "w") as fi:
        fi.write_mesh(mesh)
        fi.write_meshtags(
            cell_tags, mesh.geometry,
            geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']/Geometry")
        fi.write_meshtags(
            facet_tags, mesh.geometry,
            geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']/Geometry")

        fi.write_mesh(wedge_mesh)
        fi.write_meshtags(
            wedge_facet_tags, wedge_mesh.geometry,
            geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{wedge_mesh.name}']/Geometry")

        fi.write_mesh(slab_mesh)
        fi.write_meshtags(
            slab_facet_tags, slab_mesh.geometry,
            geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{slab_mesh.name}']/Geometry")