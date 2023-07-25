import gmsh
import dolfinx
from dolfinx import io, mesh
from mpi4py import MPI
import numpy as np
import basix.ufl
import ufl


volume_id = {"fluid": 1}
boundary_id = {"core": 2, "surface": 3}


def generate_earth(comm, gdim, r0, r1, h=0.1, geom_p=1):
    gmsh.initialize()

    if comm.rank == 0:
        gmsh.model.add("earth_shell")
        factory = gmsh.model.occ

        sph0 = factory.addSphere(0.0, 0.0, 0.0, r0)
        sph1 = factory.addSphere(0.0, 0.0, 0.0, r1)
        shell = factory.cut([(3, sph1)], [(3, sph0)],
                            removeTool=True, removeObject=True)[0]
        factory.synchronize()
        surfs = gmsh.model.getBoundary(shell)

        gmsh.model.addPhysicalGroup(3, [shell[0][1]], volume_id["fluid"])
        gmsh.model.addPhysicalGroup(2, [surfs[0][1]], boundary_id["surface"])
        gmsh.model.addPhysicalGroup(2, [surfs[1][1]], boundary_id["core"])

        gmsh.option.set_number("Mesh.CharacteristicLengthMax", h)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.set_order(geom_p)

    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.none)
    msh, cell_tags, ft = io.gmshio.model_to_mesh(
        gmsh.model, comm, rank=0, gdim=gdim, partitioner=partitioner)
    ft.name = "Facet markers"

    return msh, ft, boundary_id


def generate_box(comm: MPI.Intracomm, coords: list, n: list, order: int,
                 gen_on_rank: int=0):

    domain = ufl.Mesh(basix.ufl.element(
        "Q", "hexahedron", order, gdim=3,
        lagrange_variant=basix.LagrangeVariant.equispaced, shape=(3,)))
    dofs_per_cell = len(
        domain.ufl_coordinate_element().entity_closure_dofs[3][0])
    num_vertices = dofs_per_cell // domain.ufl_coordinate_element().block_size

    cells = np.empty((0, num_vertices), dtype=int)
    geom = np.empty((0, 3), dtype=np.double)
    if comm.rank == gen_on_rank:
        p0, p1 = coords
        nx, ny, nz = n
        n_cells = nx * ny * nz

        pts = np.stack(list(map(np.ravel,
                                np.meshgrid(
                                    np.linspace(p0[0], p1[0], order * nx + 1),
                                    np.linspace(p0[1], p1[1], order * ny + 1),
                                    np.linspace(p0[2], p1[2], order * nz + 1))))).T
        idx = np.lexsort([pts[:, i] for i in range(3)])
        geom = pts[idx]

        np_x, np_y, np_z = (order * nx + 1), (order * ny + 1), (order * nz + 1)
        n_pts = np_x * np_y * np_z

        cells = []
        for i in range(n_cells):
            iz = i // (nx * ny)
            j = i % (nx * ny)
            iy = j // (nx)
            ix = j % (nx)

            v0 = (iz * (ny + 1) + iy) * (nx + 1) + ix  # Corner of cell
            v1 = v0 + 1  # Move along x
            v2 = v0 + (nx + 1)  # Move into y
            v3 = v1 + (nx + 1)
            v4 = v0 + (nx + 1) * (ny + 1)  # Move into z
            v5 = v1 + (nx + 1) * (ny + 1)
            v6 = v2 + (nx + 1) * (ny + 1)
            v7 = v3 + (nx + 1) * (ny + 1)

            v0 = order * iz * np_y * np_x + order * iy * np_x + ix * order

            def coord_to_vertex(x, y, z):
                return v0 + x + np_x * y + np_x * np_y * z

            cell = [coord_to_vertex(x, y, z) for x, y, z in [
                (0, 0, 0), (order, 0, 0), (0, order, 0), (order, order, 0),
                (0, 0, order), (order, 0, order), (0, order, order),
                (order, order, order)]]

            if order > 1:
                for i in range(1, order):
                    cell.append(coord_to_vertex(i, 0, 0))
                for i in range(1, order):
                    cell.append(coord_to_vertex(0, i, 0))
                for i in range(1, order):
                    cell.append(coord_to_vertex(0, 0, i))
                for i in range(1, order):
                    cell.append(coord_to_vertex(order, i, 0))
                for i in range(1, order):
                    cell.append(coord_to_vertex(order, 0, i))
                for i in range(1, order):
                    cell.append(coord_to_vertex(i, order, 0))
                for i in range(1, order):
                    cell.append(coord_to_vertex(0, order, i))
                for i in range(1, order):
                    cell.append(coord_to_vertex(order, order, i))
                for i in range(1, order):
                    cell.append(coord_to_vertex(i, 0, order))
                for i in range(1, order):
                    cell.append(coord_to_vertex(0, i, order))
                for i in range(1, order):
                    cell.append(coord_to_vertex(order, i, order))
                for i in range(1, order):
                    cell.append(coord_to_vertex(i, order, order))

                for j in range(1, order):
                    for i in range(1, order):
                        cell.append(coord_to_vertex(i, j, 0))
                for j in range(1, order):
                    for i in range(1, order):
                        cell.append(coord_to_vertex(i, 0, j))
                for j in range(1, order):
                    for i in range(1, order):
                        cell.append(coord_to_vertex(0, i, j))
                for j in range(1, order):
                    for i in range(1, order):
                        cell.append(coord_to_vertex(order, i, j))
                for j in range(1, order):
                    for i in range(1, order):
                        cell.append(coord_to_vertex(i, order, j))
                for j in range(1, order):
                    for i in range(1, order):
                        cell.append(coord_to_vertex(i, j, order))

                for k in range(1, order):
                    for j in range(1, order):
                        for i in range(1, order):
                            cell.append(coord_to_vertex(i, j, k))

            if order == 1:
                cell_re = np.array([v0, v1, v2, v3, v4, v5, v6, v7])
                assert np.all(np.array(cell) == cell_re)
            cells.append(cell)

    mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, geom, domain)
    return mesh


def generate_sectant_cap(comm: MPI.Intracomm, r0: float, r1: float, n: list,
                         order: int):
    # reference coords to generate the cap
    coords = [[-1, -1, 0], [1, 1, 1]]
    mesh = generate_box(comm=comm, coords=coords, n=n, order=order)

    core_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[2], 0.0))
    core_labels = np.full_like(
        core_facets, boundary_id["core"], dtype=np.int32)
    surf_facets = dolfinx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[2], 1.0))
    surf_labels = np.full_like(
        surf_facets, boundary_id["surface"], dtype=np.int32)

    bdry_facets = np.concatenate((core_facets, surf_facets))
    bdry_labels = np.concatenate((core_labels, surf_labels))
    idx_order = np.argsort(bdry_facets)
    fts = dolfinx.mesh.meshtags(
        mesh, mesh.topology.dim - 1,
        bdry_facets[idx_order], bdry_labels[idx_order])

    def mapoct(xi):
        x = np.zeros_like(xi)
        x[:,2] = (r1/r0)**(xi[:,2] - 1.0) / np.sqrt(
            np.tan(np.pi * xi[:,0]/4.0)**2
            + np.tan(np.pi * xi[:,1]/4.0)**2
            + 1)
        x[:,0] = x[:,2] * np.tan(np.pi*xi[:,0] / 4.0)
        x[:,1] = x[:,2] * np.tan(np.pi*xi[:,1] / 4.0)
        return x

    # mesh.geometry.x[:] = sph2card(mesh.geometry.x)
    mesh.geometry.x[:] = mapoct(mesh.geometry.x)
    mesh.geometry.x[:] = mesh.geometry.x * r1

    return mesh, fts, boundary_id


if __name__ == "__main__":
    mesh, _, _ = generate_sectant_cap(MPI.COMM_WORLD, 1.0, 2.0, [2, 2, 2], 2)
    with dolfinx.io.VTXWriter(mesh.comm, "mesh.bp", mesh) as fi:
        fi.write(0.0)