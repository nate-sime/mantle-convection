from mpi4py import MPI
import enum
import numpy as np
import gmsh
import dolfinx


class Labels(enum.IntEnum):
    outer_face = 1
    inner_face = 2
    volume = 3
    cyl_top_face = 4
    cyl_bot_face = 5
    cyl_ext_face = 6

def generate_disk(comm: MPI.Intracomm,
                  geom_degree: int = 1,
                  gmsh_opts: dict[[str, [str, int, float]]] = {}):
    gmsh.initialize()
    if comm.rank == 0:
        gmsh.model.add("disc")

        inner_disk = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, 1.0, 1.0)
        outer_disk = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, 2.0, 2.0)
        annulus = gmsh.model.occ.cut([(2, outer_disk)], [(2, inner_disk)],
                                     removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()

        volumes = gmsh.model.getEntities(2)
        gmsh.model.addPhysicalGroup(2, [v[1] for v in volumes], tag=Labels.volume)
        gmsh.model.addPhysicalGroup(1, [2], tag=Labels.inner_face)
        gmsh.model.addPhysicalGroup(1, [1], tag=Labels.outer_face)

        for opt_key, opt_val in gmsh_opts.items():
            gmsh.option.setNumber(opt_key, opt_val)

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.set_order(geom_degree)

    partitioner = dolfinx.mesh.create_cell_partitioner(
        dolfinx.mesh.GhostMode.none)
    mesh, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, rank=0, gdim=2, partitioner=partitioner)

    gmsh.finalize()
    return mesh, cell_tags, facet_tags


def generate_cylinder(comm: MPI.Intracomm,
                      geom_degree: int = 1,
                      gmsh_opts: dict[[str, [str, int, float]]] = {}):
    gmsh.initialize()
    if comm.rank == 0:
        gmsh.model.add("cylinder")

        dz = 1
        r0 = 0.25
        r1 = 2.0
        cylinder1 = gmsh.model.occ.addCylinder(
            0, 0, 0, 0, 0, dz, r1)
        cylinder0 = gmsh.model.occ.addCylinder(
            0, 0, -dz, 0, 0, 2*dz, r0)
        annulus = gmsh.model.occ.cut([(3, cylinder1)], [(3, cylinder0)],
                                     removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()

        volumes = gmsh.model.getEntities(3)
        gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes], tag=Labels.volume)

        surfs = gmsh.model.getEntities(2)
        coms = np.array([
            gmsh.model.occ.getCenterOfMass(2, s[1]) for s in surfs],
            dtype=np.float64)
        top_idx = surfs[np.argmax(coms[:,2])][1]
        bot_idx = surfs[np.argmin(coms[:,2])][1]
        ext_idx = list(set(s[1] for s in surfs) - set([top_idx, bot_idx]))

        gmsh.model.addPhysicalGroup(2, [top_idx], tag=Labels.cyl_top_face)
        gmsh.model.addPhysicalGroup(2, [bot_idx], tag=Labels.cyl_bot_face)
        gmsh.model.addPhysicalGroup(2, ext_idx, tag=Labels.cyl_ext_face)

        for opt_key, opt_val in gmsh_opts.items():
            gmsh.option.setNumber(opt_key, opt_val)

        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.set_order(geom_degree)

    partitioner = dolfinx.mesh.create_cell_partitioner(
        dolfinx.mesh.GhostMode.none)
    mesh, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, rank=0, gdim=3, partitioner=partitioner)

    gmsh.finalize()
    return mesh, cell_tags, facet_tags


if __name__ == "__main__":
    # mesh_name = "disk"
    # mesh, cell_tags, facet_tags = generate_disk(
    #     MPI.COMM_WORLD, geom_degree=2, gmsh_opts={
    #         "Mesh.RecombinationAlgorithm": 2,
    #         "Mesh.RecombineAll": 2,
    #         "Mesh.CharacteristicLengthFactor": 0.1
    #     })

    mesh_name = "cylinder"
    mesh, cell_tags, facet_tags = generate_cylinder(
        MPI.COMM_WORLD, geom_degree=1, gmsh_opts={
            "Mesh.CharacteristicLengthFactor": 0.25
        })

    cell_tags.name = "cells"
    facet_tags.name = "facets"
    mesh.name = "mesh"

    with dolfinx.io.XDMFFile(mesh.comm, f"{mesh_name}.xdmf", "w") as fi:
        fi.write_mesh(mesh)
        fi.write_meshtags(
            cell_tags, mesh.geometry,
            geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']/Geometry")
        fi.write_meshtags(
            facet_tags, mesh.geometry,
            geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']/Geometry")