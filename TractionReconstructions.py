import numpy as np

def CreateTractionFunction(YoungsModulus,PoissonRatio,Model):
    E = YoungsModulus
    Nu = PoissonRatio
    lmbda = E*Nu/((1+Nu)*(1-2*Nu))
    mu = E/(2*(1+Nu))
    if Model == "LinearElasticity":
        def epsilon(u):
            return ufl.sym(ufl.grad(u))
        def sigma(u):
            return lmbda * ufl.tr(epsilon(u)) * ufl.Identity(3) + 2.0 * mu * epsilon(u)
    return sigma

def CalculateTractionMap(ObjectMesh,StressFunction,DisplacementFunction):
    V = dolfinx.fem.functionspace(ObjectMesh, ("DG", 0, (3,)))
    ds = ufl.Measure("ds", domain=ObjectMesh, subdomain_data=bound)
    Normal = ufl.FacetNormal(ObjectMesh)
    v = ufl.TestFunction(VQ)

    TractionExpr = ufl.dot(sigma(DisplacementFunction), Normal)
    TractionForm = fem.form(ufl.inner(TractionExpr, v) * ds)
    TractionFunc = fem.Function(V)
    TractionVec = fem.petsc.assemble_vector(TractionForm)
    TractionVec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    TractionFunc.x.array[:] = TractionVec.array[:]
    TractionFunc.x.scatter_forward()
    return TractionFunc

def ExportTractionFunctionsXDMF(Name,ObjectMesh,TractionFieldRegularised,TractionFieldExperimental):
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, Name+"OutputTractionExperimental.xdmf", "w") as xdmf_outfile:
        xdmf_outfile.write_mesh(ObjectMesh)
        TractionFieldExperimental.name = "Traction"
        xdmf_outfile.write_function(TractionFieldExperimental)
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, Name+"OutputTractionRegularised.xdmf", "w") as xdmf_outfile:
        xdmf_outfile.write_mesh(ObjectMesh)
        TractionFieldRegularised.name = "Traction"
        xdmf_outfile.write_function(TractionFieldRegularised)

def RegulariseDisplacementField():
    pass
