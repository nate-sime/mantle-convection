import pathlib

import numpy as np
import dolfinx
import geomdl.fitting
import geomdl.abstract
import geomdl.exchange
from mpi4py import MPI

from sz.mesh_utils import extract_submesh_and_transfer_facets
from sz.model import Labels

import mesh_generator2d


def plot_spline_matplotlib(spline, nu=64):
    import matplotlib.pyplot as plt
    u = np.linspace(0.0, 1.0, nu)
    x = np.array(spline.evaluate_list(u), dtype=np.float64)
    plt.plot(x[:,0], x[:,1], "-")
    plt.show()


def generate_mesh_step(comm: MPI.Intracomm,
                       slab_spline: geomdl.abstract.Curve,
                       file_path: str | pathlib.Path):
    mesh, cell_tags, facet_tags = mesh_generator2d.generate(
        comm, slab_spline, wedge_x_buffer, plate_y,
        corner_resolution, surface_resolution, bulk_resolution,
        couple_y=couple_y)
    cell_tags.name = "zone_cells"
    facet_tags.name = "zone_facets"
    mesh.name = "zone"

    wedge_mesh, wedge_facet_tags = extract_submesh_and_transfer_facets(
        mesh, facet_tags, cell_tags.indices[cell_tags.values == Labels.wedge])
    wedge_facet_tags.name = "wedge_facets"
    wedge_mesh.name = "wedge"
    slab_mesh, slab_facet_tags = extract_submesh_and_transfer_facets(
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

        fi.write_mesh(slab_mesh)
        fi.write_meshtags(
            slab_facet_tags, slab_mesh.geometry,
            geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{slab_mesh.name}']/Geometry")


if __name__ == "__main__":
    slab_spline_degree = 3

    t_final_yr = 11e6
    n_slab_steps = 1

    depth = 600.0
    x_slab = np.linspace(0, depth, 10)
    wedge_x_buffer = 50.0
    plate_y = -50.0
    couple_y = None
    bulk_resolution = 25.0
    corner_resolution = 2.0
    surface_resolution = 5.0

    # The initial slab surface
    y_slab = -x_slab
    slab_spline_t0 = geomdl.fitting.interpolate_curve(
        np.stack((x_slab, y_slab)).T.tolist(), degree=slab_spline_degree)

    # The final slab surface
    y_slab = -depth**-1 * x_slab**2
    slab_spline_tfinal = geomdl.fitting.interpolate_curve(
        np.stack((x_slab, y_slab)).T.tolist(), degree=slab_spline_degree)

    # The time dependent slab
    slab_spline = geomdl.BSpline.Curve()
    slab_spline.degree = slab_spline_t0.degree
    slab_spline.ctrlpts = slab_spline_t0.ctrlpts
    slab_spline.knotvector = slab_spline_t0.knotvector

    def ctrl_pt_transfer_fn(theta):
        ctrl0 = np.array(slab_spline_t0.ctrlpts, dtype=np.float64)
        ctrl1 = np.array(slab_spline_tfinal.ctrlpts, dtype=np.float64)
        return (theta * ctrl1 + (1 - theta) * ctrl0).tolist()


    idx_fmt = f"0{int(np.ceil(np.log10(n_slab_steps + 1)))}"
    directory = pathlib.Path("evolving2d")
    for i, t_yr in enumerate(np.linspace(0.0, t_final_yr, n_slab_steps + 1)):
        print(f"Generating mesh {i}: t = {t_yr:.3e} yr")
        slab_spline.ctrlpts = ctrl_pt_transfer_fn(theta=t_yr / t_final_yr)
        generate_mesh_step(MPI.COMM_SELF, slab_spline,
                           directory / f"subduction_zone_{i:{idx_fmt}}.xdmf")

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



