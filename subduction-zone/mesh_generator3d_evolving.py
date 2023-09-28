import pathlib
import typing

import numpy as np
import dolfinx
import geomdl.fitting
import geomdl.abstract
import geomdl.exchange
from mpi4py import MPI

from sz.mesh_utils import extract_submesh_and_transfer_meshtags
from sz.model import Labels

import mesh_generator3d


def generate_mesh_step(comm: MPI.Intracomm,
                       slab_spline: geomdl.abstract.Surface,
                       slab_spline_bbox: typing.Sequence[
                           typing.Sequence[float]],
                       slab_dz: float,
                       file_path: str | pathlib.Path):
    """
    Given a spline surface definition, call the appropriate mesh generation
    routine and write data to file.

    Args:
        comm: MPI communicator
        slab_spline: Spline defining the slab interface
        slab_spline_bbox: The bounding box of the spline surface in the
         form `[[xmin, ymin, zmin], [xmax, ymax, zmax]]`
        slab_dz: Distance in the z direction by which to extrude the slab
         surface interface to generate the slab volume
        file_path: Output mesh path
    """
    mesh, cell_tags, facet_tags = mesh_generator3d.generate(
        comm, slab_spline, slab_spline_bbox, plate_y, slab_dz,
        corner_resolution, surface_resolution, bulk_resolution,
        couple_y=couple_y)
    cell_tags.name = "zone_cells"
    facet_tags.name = "zone_facets"
    mesh.name = "zone"

    wedge_mesh, wedge_facet_tags = extract_submesh_and_transfer_meshtags(
        mesh, facet_tags, cell_tags.indices[cell_tags.values == Labels.wedge])
    wedge_facet_tags.name = "wedge_facets"
    wedge_mesh.name = "wedge"

    wedgeplate_mesh, (wedgeplate_facet_tags, wedgeplate_cell_tags) = \
        extract_submesh_and_transfer_meshtags(
            mesh, [facet_tags, cell_tags], cell_tags.indices[
                np.isin(cell_tags.values, (Labels.wedge, Labels.plate))])
    wedgeplate_facet_tags.name = "wedgeplate_facets"
    wedgeplate_cell_tags.name = "wedgeplate_cells"
    wedgeplate_mesh.name = "wedgeplate"

    slab_mesh, slab_facet_tags = extract_submesh_and_transfer_meshtags(
        mesh, facet_tags, cell_tags.indices[cell_tags.values == Labels.slab])
    slab_facet_tags.name = "slab_facets"
    slab_mesh.name = "slab"

    with dolfinx.io.XDMFFile(mesh.comm, file_path, "w") as fi:
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

        fi.write_mesh(wedgeplate_mesh)
        fi.write_meshtags(
            wedgeplate_facet_tags, wedgeplate_mesh.geometry,
            geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{wedgeplate_mesh.name}']/Geometry")
        fi.write_meshtags(
            wedgeplate_cell_tags, wedgeplate_mesh.geometry,
            geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{wedgeplate_mesh.name}']/Geometry")

        fi.write_mesh(slab_mesh)
        fi.write_meshtags(
            slab_facet_tags, slab_mesh.geometry,
            geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{slab_mesh.name}']/Geometry")


if __name__ == "__main__":
    slab_spline_degree = 3

    # Initial surface geometry
    x_slab, y_slab, z_slab = mesh_generator3d.straight_dip(
        depth=300.0, yrange=[-438.8, 438.8])
    slab_xyz = np.vstack((x_slab.ravel(), y_slab.ravel(), z_slab.ravel())).T
    slab_xyz = slab_xyz[np.lexsort((slab_xyz[:,  0], slab_xyz[:, 1]))]

    import geomdl.fitting
    slab_xy_shape = [x_slab.shape[0], y_slab.shape[1]]
    slab_spline_degree = 3
    slab_spline_t0 = geomdl.fitting.interpolate_surface(
        slab_xyz.tolist(), slab_xy_shape[0], slab_xy_shape[1],
        degree_u=slab_spline_degree, degree_v=slab_spline_degree)

    # Final surface geometry
    x_slab, y_slab, z_slab = mesh_generator3d.mariana_like()
    slab_xyz = np.vstack(
        (x_slab.ravel(), y_slab.ravel(), z_slab.ravel())).T
    slab_xyz = slab_xyz[np.lexsort((slab_xyz[:, 0], slab_xyz[:, 1]))]
    slab_xy_shape = [x_slab.shape[0], y_slab.shape[1]]
    slab_spline_degree = 3
    slab_spline_tfinal = geomdl.fitting.interpolate_surface(
        slab_xyz.tolist(), slab_xy_shape[0], slab_xy_shape[1],
        degree_u=slab_spline_degree, degree_v=slab_spline_degree)

    t_final_yr = 11e6
    n_slab_steps = 10

    plate_y = -50.0
    slab_dz = -200.0
    couple_y = plate_y - 10.0
    bulk_resolution = 100.0
    corner_resolution = 10.0
    surface_resolution = 20.0

    # The time dependent slab
    slab_spline = geomdl.BSpline.Surface()
    slab_spline.ctrlpts_size_u = slab_spline_t0.ctrlpts_size_u
    slab_spline.ctrlpts_size_v = slab_spline_t0.ctrlpts_size_v
    slab_spline.degree = slab_spline_t0.degree
    slab_spline.ctrlpts = slab_spline_t0.ctrlpts
    slab_spline.knotvector = slab_spline_t0.knotvector

    def ctrl_pt_transfer_fn(theta):
        ctrl0 = np.array(slab_spline_t0.ctrlpts, dtype=np.float64)
        ctrl1 = np.array(slab_spline_tfinal.ctrlpts, dtype=np.float64)
        return (theta * ctrl1 + (1 - theta) * ctrl0).tolist()


    # Generate meshes in time loop
    idx_fmt = f"0{int(np.ceil(np.log10(n_slab_steps + 1)))}"
    directory = pathlib.Path("evolving3d")
    for i, t_yr in enumerate(np.linspace(0.0, t_final_yr, n_slab_steps + 1)):
        print(f"Generating mesh {i}: t = {t_yr:.3e} yr")
        slab_spline.ctrlpts = ctrl_pt_transfer_fn(theta=t_yr / t_final_yr)
        slab_spline_bbox = mesh_generator3d.compute_spline_bbox(
            slab_spline, [0.55, 0.55])
        generate_mesh_step(
            MPI.COMM_SELF, slab_spline, slab_spline_bbox, slab_dz,
            directory / f"subduction_zone_{i:{idx_fmt}}.xdmf")

    # Write metadata: simulation time, number of slab deformation steps (i.e.
    # time steps), the mesh files index format and the spline data. These
    # metadata are read in by the finite element simulation script.
    meta_data = {
        "t_final_yr": t_final_yr,
        "n_slab_steps": n_slab_steps,
        "idx_fmt": idx_fmt,
    }
    import json
    with open(directory / "metadata.json", "w") as fi:
        json.dump(meta_data, fi, indent=2)

    geomdl.exchange.export_json(
        slab_spline_t0, directory / "slab_spline_t0.json")
    geomdl.exchange.export_json(
        slab_spline_tfinal, directory / "slab_spline_tfinal.json")



