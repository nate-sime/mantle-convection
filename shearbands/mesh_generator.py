from mpi4py import MPI
import enum
import gmsh
import dolfinx


class Labels(enum.IntEnum):
    outer_face = 1
    inner_face = 2
    volume = 3

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


if __name__ == "__main__":
    mesh, cell_tags, facet_tags = generate_disk(
        MPI.COMM_WORLD, geom_degree=2, gmsh_opts={
            "Mesh.RecombinationAlgorithm": 2,
            "Mesh.RecombineAll": 2,
            "Mesh.CharacteristicLengthFactor": 0.1
        })

    cell_tags.name = "cells"
    facet_tags.name = "facets"
    mesh.name = "mesh"

    with dolfinx.io.XDMFFile(mesh.comm, "disk.xdmf", "w") as fi:
        fi.write_mesh(mesh)
        fi.write_meshtags(
            cell_tags, mesh.geometry,
            geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']/Geometry")
        fi.write_meshtags(
            facet_tags, mesh.geometry,
            geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']/Geometry")