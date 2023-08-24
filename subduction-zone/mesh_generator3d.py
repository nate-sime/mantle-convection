import typing

import geomdl
import geomdl.abstract
import dolfinx
from mpi4py import MPI
import gmsh
import numpy as np

import mesh_generator

Labels = mesh_generator.Labels


def plot_spline_surface_pyvista(spline):
    import pyvista
    u, v = np.mgrid[0:1:20j, 0:1:20j]
    uv = np.c_[u.ravel(), v.ravel()]
    X = np.array(spline.evaluate_list(uv), dtype=np.float64)
    pyvista.PolyData(X).delaunay_2d().plot(scalars=X[:, 2], show_bounds=True)


def generate(comm: MPI.Intracomm,
             slab_xyz: typing.Iterable[typing.Iterable[float]],
             slab_xy_shape: typing.Iterable[int],
             plate_y: float,
             slab_dz: float,
             corner_resolution: float,
             surface_resolution: float,
             bulk_resolution: float,
             couple_y: float = None,
             slab_spline_degree: int = None,
             slab_spline: geomdl.abstract.Curve = None,
             geom_degree: int = 1):
    gmsh.initialize()
    if comm.rank == 0:
        if slab_spline_degree is None:
            slab_spline_degree = 3

        if not isinstance(slab_xyz, np.ndarray):
            slab_xyz = np.array(slab_xyz, dtype=np.float64)

        if slab_spline is None:
            import geomdl.fitting
            slab_spline = geomdl.fitting.interpolate_surface(
                slab_xyz.tolist(), slab_xy_shape[0], slab_xy_shape[1],
                degree_u=slab_spline_degree, degree_v=slab_spline_degree)

        plot_spline_surface_pyvista(slab_spline)

        slab_width = slab_xyz[:,0].max()
        slab_length = slab_xyz[:,1].max()
        slab_depth = slab_xyz[:,2].min()
        slab_x0 = [slab_xyz[:,0].min(), slab_xyz[:,1].min(), slab_xyz[:,2].max()]

        gmsh.model.add("subduction")

        # -- Geometry definition
        cpts = [gmsh.model.occ.addPoint(*pt) for pt in slab_spline.ctrlpts]
        interface = gmsh.model.occ.addBSplineSurface(
            cpts, slab_spline.ctrlpts_size_u, degreeU=slab_spline.degree_u,
            degreeV=slab_spline.degree_v)
        gmsh.model.occ.synchronize()

        slab = gmsh.model.occ.extrude(
            [(2, interface)], dx=0, dy=0, dz=slab_dz)
        wedge = gmsh.model.occ.extrude(
            [(2, interface)], dx=0, dy=0, dz=-slab_depth * 1.1)

        wires = gmsh.model.getBoundary([(2, interface)], oriented=False)
        coms = np.array([gmsh.model.occ.getCenterOfMass(1, w[1]) for w in wires])
        top_wire = wires[np.argmax(coms[:,2])]
        surface_to_cut = gmsh.model.occ.extrude(
            [top_wire], dx=slab_width*1.1, dy=0, dz=0)
        gmsh.model.occ.synchronize()

        wedge_shapes, _ = gmsh.model.occ.fragment(
            [w for w in wedge if w[0] == 3],
            [s for s in surface_to_cut if s[0] == 2],
            removeObject=True, removeTool=True)

        gmsh.model.occ.synchronize()
        wedge_vols = [v for v in wedge_shapes if v[0] == 3]
        wedge_surfs = [v for v in wedge_shapes if v[0] == 2]
        coms = np.array([gmsh.model.occ.getCenterOfMass(3, v[1])
                         for v in wedge_vols])
        gmsh.model.occ.remove([wedge_vols[np.argmax(coms[:, 2])]],
                              recursive=True)
        coms = np.array([gmsh.model.occ.getCenterOfMass(2, v[1])
                         for v in wedge_surfs])
        gmsh.model.occ.remove([wedge_surfs[np.argmax(coms[:, 0])]],
                              recursive=True)
        gmsh.model.occ.synchronize()

        # if couple_y is not None:
        #     assert couple_y < plate_y
        #     couple_surf = gmsh.model.occ.addRectangle(
        #         slab_x0[0], slab_x0[1], couple_y, slab_width, slab_length)

        # Generate delimiting horizontal lines of constant depth. Particularly
        # useful for depth dependent material coefficients
        def fragment_wedge_with_horizontal_surface(surface):
            vols = gmsh.model.occ.getEntities(3)
            coms = np.array([gmsh.model.occ.getCenterOfMass(*v) for v in vols])
            wedge = vols[np.argmax(coms[:,0])]
            slab = vols[np.argmin(coms[:,0])]
            new_surf, _ = gmsh.model.occ.intersect(
                [wedge], [(2, surface)], removeTool=True, removeObject=False)
            gmsh.model.occ.fragment(
                [wedge, slab], new_surf, removeObject=True, removeTool=True)
            gmsh.model.occ.remove([(2, surface)], recursive=True)
            gmsh.model.occ.synchronize()

        plate_iface = gmsh.model.occ.addRectangle(
            slab_x0[0], slab_x0[1], plate_y, slab_width, slab_length)
        fragment_wedge_with_horizontal_surface(plate_iface)

        # -- Labelling
        # Volume labels
        vols = gmsh.model.occ.getEntities(3)
        coms = np.array([gmsh.model.occ.getCenterOfMass(*v) for v in vols])
        plate = vols[np.argmax(coms[:,2])]
        slab = vols[np.argmin(coms[:,0])]
        wedge = list(set(vols) - set([plate, slab]))

        gmsh.model.addPhysicalGroup(3, [plate[1]], tag=Labels.plate)
        gmsh.model.addPhysicalGroup(3, [slab[1]], tag=Labels.slab)
        gmsh.model.addPhysicalGroup(3, [w[1] for w in wedge], tag=Labels.wedge)

        # Domain interface facet labels
        plate_facets = gmsh.model.getAdjacencies(plate[0], plate[1])[1]
        slab_facets = gmsh.model.getAdjacencies(slab[0], slab[1])[1]
        wedge_facets = np.fromiter(set(sum((
            gmsh.model.getAdjacencies(3, w[1])[1].tolist() for w in wedge), [])
        ), dtype=np.int32)
        free_slip_faces = []

        slab_plate = np.intersect1d(slab_facets, plate_facets)
        slab_wedge = np.intersect1d(slab_facets, wedge_facets)
        plate_wedge = np.intersect1d(plate_facets, wedge_facets)
        gmsh.model.addPhysicalGroup(2, slab_plate, tag=Labels.slab_plate)
        gmsh.model.addPhysicalGroup(2, slab_wedge, tag=Labels.slab_wedge)
        gmsh.model.addPhysicalGroup(2, plate_wedge, tag=Labels.plate_wedge)

        # Domain exterior facet labels
        def facet_coms(facets):
            return np.array([gmsh.model.occ.getCenterOfMass(2, f) for f in facets])
        coms_plate = facet_coms(plate_facets)
        coms_wedge = facet_coms(wedge_facets)
        coms_slab = facet_coms(slab_facets)

        # slab
        slab_left_face = [slab_facets[np.argmin(coms_slab[:, 0])]]
        gmsh.model.addPhysicalGroup(2, slab_left_face, tag=Labels.slab_left)
        slab_right_face = [slab_facets[np.argmin(coms_slab[:, 2])]]
        gmsh.model.addPhysicalGroup(2, slab_right_face, tag=Labels.slab_right)
        # find the remaining face that's not shared
        unwanted_faces = set.union(
            *map(set, (
                slab_left_face, slab_right_face, wedge_facets, plate_facets)))
        remaining_slab_facets = list(set(slab_facets) - unwanted_faces)
        coms = np.array([gmsh.model.occ.getCenterOfMass(2, s) for s in remaining_slab_facets])
        slab_bottom_face = remaining_slab_facets[np.argmin(coms[:, 2])]
        gmsh.model.addPhysicalGroup(2, [slab_bottom_face], tag=Labels.slab_bottom)

        free_slip_faces += list(set(remaining_slab_facets) - set([slab_bottom_face]))

        # wedge
        wedge_right_face = [wedge_facets[np.argmax(coms_wedge[:, 0])]]
        gmsh.model.addPhysicalGroup(2, wedge_right_face, tag=Labels.wedge_right)
        unwanted_faces = set.union(
            *map(set, (wedge_right_face, slab_facets,  plate_facets)))
        free_slip_faces += list(set(wedge_facets) - unwanted_faces)

        # plate
        gmsh.model.addPhysicalGroup(2,
                                    [plate_facets[np.argmax(coms_plate[:, 0])]],
                                    tag=Labels.plate_right)
        gmsh.model.addPhysicalGroup(2,
                                    [plate_facets[np.argmax(coms_plate[:, 2])]],
                                    tag=Labels.plate_top)

        # free slip
        gmsh.model.addPhysicalGroup(2, free_slip_faces, tag=Labels.free_slip)

        # -- Local refinement
        def get_codim_entities(dimtag):
            if isinstance(dimtag, tuple):
                dimtag = [dimtag]
            return gmsh.model.getBoundary(dimtag, oriented=False, recursive=False)

        # Apply local refinement to the coupling point if it exists, otherwise
        # the wedge corner
        if couple_y is None:
            corner_pt = set.intersection(
                *map(lambda x: set(get_codim_entities(x)), (plate, slab, wedge)))
            assert len(corner_pt) == 1
            quit()
            corner_pt = corner_pt.pop()
        else:
            surfs = list(map(get_codim_entities, (slab, wedge)))
            common_surfs = set.intersection(*map(set, surfs))
            wires = list(set(*map(get_codim_entities, list(common_surfs))))
            coms = np.array(list(gmsh.model.occ.getCenterOfMass(1, w[1]) for w in wires))
            corner_wire = wires[np.argmin(np.abs(coms[:, 2] - couple_y))]

        corner_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(corner_dist, "CurvesList", [corner_wire[1]])

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
            (gmsh.model.getEntitiesForPhysicalGroup(2, Labels.slab_wedge),
             gmsh.model.getEntitiesForPhysicalGroup(2, Labels.plate_wedge)))
        gmsh.model.mesh.field.setNumbers(interface_dist, "FacesList", refine_surfs)
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

        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.set_order(geom_degree)

    partitioner = dolfinx.mesh.create_cell_partitioner(
        dolfinx.mesh.GhostMode.none)
    mesh, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, rank=0, gdim=3, partitioner=partitioner)

    gmsh.finalize()
    return mesh, cell_tags, facet_tags


if __name__ == "__main__":
    from mpi4py import MPI

    x_slab, y_slab = np.meshgrid(
        np.linspace(0, 300.0, 10), np.linspace(0, 300.0, 10))
    z_slab = -x_slab
    sli = slice(2,-3)
    z_slab[sli,sli] += np.random.normal(np.random.normal(size=z_slab[sli,sli].shape, scale=1.0)*10)
    slab_xyz = np.vstack((x_slab.ravel(), y_slab.ravel(), z_slab.ravel())).T
    slab_xyz = slab_xyz[np.lexsort((slab_xyz[:,  0], slab_xyz[:, 1]))]

    plate_y = -50.0
    slab_dz = -200.0
    couple_y = plate_y - 10.0 #None
    bulk_resolution = 100.0
    corner_resolution = 10.0
    surface_resolution = 20.0

    mesh, cell_tags, facet_tags = generate(
        MPI.COMM_WORLD, slab_xyz,
        [x_slab.shape[0], y_slab.shape[1]], plate_y, slab_dz,
        corner_resolution, surface_resolution, bulk_resolution,
        couple_y=couple_y, slab_spline_degree=3)
    cell_tags.name = "zone_cells"
    facet_tags.name = "zone_facets"
    mesh.name = "zone"

    wedge_mesh, wedge_facet_tags = mesh_generator.extract_submesh_and_transfer_facets(
        mesh, facet_tags, cell_tags.indices[cell_tags.values == Labels.wedge])
    wedge_facet_tags.name = "wedge_facets"
    wedge_mesh.name = "wedge"
    slab_mesh, slab_facet_tags = mesh_generator.extract_submesh_and_transfer_facets(
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