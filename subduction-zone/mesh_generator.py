import gmsh
import numpy as np


def generate_benchmark_mesh_from_surface():
    EPS = 1e-10

    gmsh.initialize()

    depth = 500.0
    dx = 50.0

    gmsh.model.add("subduction")
    domain = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, depth + dx, -depth)
    gmsh.model.occ.synchronize()
    # vertices = gmsh.model.getBoundary([(2, domain)], recursive=True)
    # pts = np.array([gmsh.model.getValue(0, e[1], []) for e in vertices])
    #
    # tl = np.where(np.isclose(pts[:,0], 0.0) & np.isclose(pts[:,1], 0.0))[0]
    # br = np.where(np.isclose(pts[:,0], depth + dx) & np.isclose(pts[:,1], -depth))[0]
    # gmsh.model.occ.addLine(tl[0]+1, br[0]+1)

    tl = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)
    br = gmsh.model.occ.addPoint(depth, -depth, 0.0)
    interface = gmsh.model.occ.addLine(tl, br)

    gmsh.model.occ.fragment([(2, domain)], [(1, interface)], removeObject=True, removeTool=True)



    gmsh.model.occ.synchronize()
    gmsh.fltk.run()


if __name__ == "__main__":
    generate_benchmark_mesh_from_surface()