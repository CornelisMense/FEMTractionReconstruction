import numpy as np
import SimpleITK as sitk
import gmsh
import meshio
import pygmsh
import scipy
import tifffile
import trimesh
from skimage import measure
from mpi4py import MPI
from dolfinx import fem, mesh, nls
from dolfinx.io import XDMFFile

def MCMesherSTL(Name,BulkField,BulkThresholdValueMin):
    PadWidth = 5
    ImageIso = zoom(BulkField,zoom=(1,1,int(Resolution_Z)),order=1)
    ImageIsoThreshold = (ImageIso>BulkThresholdValueMin)
    padded = np.pad(ImageIsoThreshold*ImageIso,pad_width=PadWidth,mode="constant",constant_values=0.0)
    verts, faces, normals, values = measure.marching_cubes(padded,level=1.0,spacing=(1.0, 1.0, Resolution_Z/int(Resolution_Z)))
    verts -= PadWidth
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh.remove_unreferenced_vertices()
    components = mesh.split(only_watertight=False)
    mesh = max(components, key=lambda m: m.area)
    mesh.export(Name+"MC.stl")
    return Name+"MC"

def ImportSTL():
    pass

def MesherXDMF(STLName):
    gmsh.initialize()
    gmsh.model.add(STLName)
    gmsh.merge(STLName+".stl")

    surfs = gmsh.model.getEntities(2)
    sl = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfs])
    vol = gmsh.model.geo.addVolume([sl])

    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(2, [sl], tag=1)
    gmsh.model.addPhysicalGroup(3, [vol], tag=1)

    h = 1.0
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)

    gmsh.model.mesh.generate(3)
    gmsh.write(STLName+".msh")
    gmsh.finalize()

    m = meshio.read(STLName+".msh")
    tet_points = m.points
    tet_tets = m.cells_dict["tetra"]
    tet_tri = m.cells_dict.get("triangle")
    meshio.write(STLName+"Volume.xdmf",meshio.Mesh(points=tet_points,cells=[("tetra", tet_tets)],cell_data={"bnd_marker": [np.zeros(len(tet_tets), dtype=np.int32)]},),)
    meshio.write(STLName+"Surface.xdmf",meshio.Mesh(points=tet_points,cells=[("triangle", tet_tri)],cell_data={"bnd_marker": [np.zeros(len(tet_tets), dtype=np.int32)]},),)

def ImportMeshXDMF(MeshName):
    with XDMFFile(MPI.COMM_WORLD, MeshName+"Volume.xdmf", 'r') as xdmf_infile:
        Mesh = xdmf_infile.read_mesh(name='Grid')
        tags = xdmf_infile.read_meshtags(Mesh, name="Grid")
    Mesh.topology.create_connectivity(Mesh.topology.dim, Mesh.topology.dim-1)

    with XDMFFile(MPI.COMM_WORLD, MeshName+"Surface.xdmf", 'r') as xdmf_infile:
        bound = xdmf_infile.read_meshtags(Mesh, name="Grid")  
    Mesh.topology.create_connectivity(Mesh.topology.dim-1, Mesh.topology.dim)
    return Mesh
