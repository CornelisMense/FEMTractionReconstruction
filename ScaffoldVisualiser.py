import numpy as np
import gmsh
import meshio
import pygmsh
from mpi4py import MPI
from petsc4py import PETSc
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from dolfinx import fem, mesh, nls
from dolfinx.io import XDMFFile

def CreateBoxMesh(MeshName,MeshResolution,MeshScale):
    warnings.filterwarnings("ignore")
    gmsh.initialize()
    gmsh.model.add("DFG 3D")
    
    Box = gmsh.model.occ.addBox(0,0,0,MeshScale[2][-1],MeshScale[1][-1],MeshScale[0][-1])
    gmsh.model.occ.synchronize()
    volumes = gmsh.model.getEntities(dim=3)
    surfaces = gmsh.model.occ.getEntities(dim=2)

    sur = []
    for surface in surfaces:
        com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
        sur.append(surface[1])
    vol = [] 
    for volume in volumes:
        vol.append(volume[1])
        
    gmsh.model.addPhysicalGroup(2, sur, 1)
    gmsh.model.setPhysicalName(2, 1, "Surface")
    gmsh.model.add_physical_group(3, vol, 1)
    gmsh.model.setPhysicalName(3, 1, "Volume")

    gmsh.model.mesh.field.add("Box", 1)
    gmsh.model.mesh.field.setNumber(1, "XMin", 0)
    gmsh.model.mesh.field.setNumber(1, "XMax", MeshScale[2][-1])
    gmsh.model.mesh.field.setNumber(1, "ZMin", 0)
    gmsh.model.mesh.field.setNumber(1, "ZMax", MeshScale[0][-1])
    gmsh.model.mesh.field.setNumber(1, "YMin", 0)
    gmsh.model.mesh.field.setNumber(1, "YMax", MeshScale[1][-1])
    gmsh.model.mesh.field.setNumber(1, "VIn", MeshResolution)
    gmsh.model.mesh.field.setNumber(1, "VOut", MeshResolution*10)
    gmsh.model.mesh.field.setNumber(1, "Thickness", MeshResolution*10)
    gmsh.model.mesh.field.setAsBackgroundMesh(1)

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(MeshName+"Box.msh")
    msh = meshio.read(MeshName+"Box.msh")

    tri_data = msh.cell_data_dict["gmsh:physical"]["tetra"]
    meshio.write(MeshName+"Box.xdmf",
        meshio.Mesh(points=msh.points,
            cells={"tetra": msh.cells_dict["tetra"]},
            cell_data={"bnd_marker": [tri_data]}
        )
    )

    line_data = msh.cell_data_dict["gmsh:physical"]["triangle"]
    meshio.write(MeshName+"BoxSurf.xdmf",
     meshio.Mesh(points=msh.points,
            cells={"triangle": msh.cells_dict["triangle"]},
            cell_data={"bnd_marker": [line_data]}
        )
    )
    gmsh.finalize()

def ReadBoxMeshXDMF(MeshName):
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, MeshName+"Box.xdmf", 'r') as xdmf_infile:
        Mesh = xdmf_infile.read_mesh(name = 'Grid')
        tags = xdmf_infile.read_meshtags(Mesh,name="Grid")
    Mesh.topology.create_connectivity(Mesh.topology.dim, Mesh.topology.dim-1)

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, MeshName+"BoxSurf.xdmf", 'r') as xdmf_infile:
       bound = xdmf_infile.read_meshtags(Mesh,name="Grid")  
    Mesh.topology.create_connectivity(Mesh.topology.dim-1, Mesh.topology.dim)
    return Mesh

def CreateIntensityInterpolator(Data,MeshScale,Type="Constant"):
    interp = RegularGridInterpolator(MeshScale, Data, bounds_error=False, fill_value=0.0,method='nearest')
    if Type=='Fade': 
        def Interpolator(x):
            pts = np.vstack((x[0], x[1], x[2])).T
            values = (MeshScale[2][-1]+5*x[2,i])*interp(pts)
            return values
    else:
        def Interpolator(x):
            pts = np.vstack((x[0], x[1], x[2])).T
            values = interp(pts)
            return values
    return Interpolator

def InterpolateIntensity(Interpolator,BoxMesh):
    V = dolfinx.fem.functionspace(BoxMesh, ("CG", 1))
    IntensityFunction = fem.Function(V)
    IntensityFunction.interpolate(Interpolator)
    return IntensityFunction

def ExportIntensityXDMF(Name,IntensityFunction,BoxMesh,AnalysisName):
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, AnalysisName+Name+"OutputI.xdmf", "w") as xdmf_outfile:
        xdmf_outfile.write_mesh(BoxMesh)
        IntensityFunction.name = "Intensity"
        xdmf_outfile.write_function(IntensityFunction)
