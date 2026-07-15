import numpy as np
import SimpleITK as sitk
from scipy.interpolate import RegularGridInterpolator
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh, nls
from dolfinx.io import XDMFFile, gmshio

def CalculateBSpline(ImageUndeformed,ImageDeformed,ResolutionZ,BSplineBasis):
    undeformed = sitk.GetImageFromArray(ImageUndeformed.astype(np.float32))
    deformed   = sitk.GetImageFromArray(ImageDeformed.astype(np.float32))
    
    undeformed.SetSpacing([1.0, 1.0, 1.0*Resolution_Z])
    deformed.SetSpacing([1.0, 1.0, 1.0*Resolution_Z])

    fixed_mask_array = (SmoothedImageBulkScaled > 0).astype(np.uint8)
    moving_mask_array = (SmoothedImageBulkScaled > 0).astype(np.uint8)
    fixed_mask = sitk.GetImageFromArray(fixed_mask_array)
    moving_mask = sitk.GetImageFromArray(moving_mask_array)
    fixed_mask.CopyInformation(undeformed)
    moving_mask.CopyInformation(deformed)
    fixed_mask_dilated  = sitk.BinaryDilate(fixed_mask,  kernelRadius=[2,2,2])
    moving_mask_dilated = sitk.BinaryDilate(moving_mask, kernelRadius=[2,2,2])

    rigid_reg = sitk.ImageRegistrationMethod()
    rigid_reg.SetMetricAsMeanSquares()
    rigid_reg.SetInterpolator(sitk.sitkLinear)
    rigid_reg.SetMetricFixedMask(fixed_mask_dilated)
    rigid_reg.SetMetricMovingMask(moving_mask_dilated)
    rigid_reg.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,numberOfIterations=50,maximumNumberOfCorrections=5,maximumNumberOfFunctionEvaluations=500,costFunctionConvergenceFactor=1e7)
    rigid_tx_init = sitk.TranslationTransform(3)
    rigid_reg.SetInitialTransform(rigid_tx_init, inPlace=False)
    rigid_transform = rigid_reg.Execute(undeformed, deformed)

    deformed_rigid_corrected = sitk.Resample(deformed,undeformed,rigid_transform.GetInverse(),sitk.sitkLinear,0.0,deformed.GetPixelID())
    moving_mask_rigid_corrected = sitk.Resample(moving_mask,undeformed,rigid_transform.GetInverse(),sitk.sitkNearestNeighbor,0,moving_mask.GetPixelID())

    image_size = undeformed.GetSize()
    mesh_size = [image_size[0]//BSplineBasis[0], image_size[1]//BSplineBasis[1], image_size[2]//BSplineBasis[2]]
    initial_transform = sitk.BSplineTransformInitializer(image1=undeformed,transformDomainMeshSize=mesh_size)

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetMetricFixedMask(fixed_mask)
    registration_method.SetMetricMovingMask(moving_mask_rigid_corrected)
    registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-6,numberOfIterations=100,maximumNumberOfCorrections=10,maximumNumberOfFunctionEvaluations=500,costFunctionConvergenceFactor=1e+7)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration_method.Execute(undeformed, deformed_rigid_corrected)

    displacement_field = sitk.TransformToDisplacementField(final_transform,sitk.sitkVectorFloat64,undeformed.GetSize(),undeformed.GetOrigin(),undeformed.GetSpacing(),undeformed.GetDirection())
    DisplacementsNumpy = sitk.GetArrayFromImage(displacement_field)
    return DisplacementsNumpy

def ExportNumpyArray(DisplacementField,Name):
    np.save(Name+'U', DisplacementField)

def ImportNumpyArray(ImportName):
    DisplacementsNumpy = np.load(ImportName)
    return DisplacementsNumpy

def CorrectDisplacementDrift(DisplacementArray,BulkImage):
    DisplacementDriftX = np.sum(DisplacementArray[:,:,:,0]*(BulkImage>0))/(np.sum(BulkImage>0))
    DisplacementDriftY = np.sum(DisplacementArray[:,:,:,1]*(BulkImage>0))/(np.sum(BulkImage>0))
    DisplacementDriftZ = np.sum(DisplacementArray[:,:,:,2]*(BulkImage>0))/(np.sum(BulkImage>0))
    DisplacementDrift = [DisplacementDriftX,DisplacementDriftY,DisplacementDriftZ]
    CorrectedDisplacementX = DisplacementArray[:,:,:,0]-DisplacementDriftX
    CorrectedDisplacementY = DisplacementArray[:,:,:,1]-DisplacementDriftY
    CorrectedDisplacementZ = DisplacementArray[:,:,:,2]-DisplacementDriftZ
    CorrectedDisplacement = np.stack((ux,uy,uz),axis=3)
    return CorrectedDisplacement, DisplacementDrift

def CreateDisplacementInterpolator(DisplacementField,MeshScale):
    Interp = RegularGridInterpolator(MeshScale, DisplacementField, bounds_error=False, fill_value=None,method='linear')
    def Interpolator(x):
        pts = np.vstack((x[0], x[1], x[2])).T
        values = Interp(pts)
        return values
    return Interpolator

def InterpolateDisplacements(Interpolator,ObjectMesh):
    V = dolfinx.fem.functionspace(ObjectMesh, ("CG", 1, (3,)))
    DisplacementFunctionExperimental = fem.Function(V)
    DisplacementFunctionExperimental.interpolate(Interpolator)
    DisplacementFunctionRegularised = fem.Function(V)
    DisplacementFunctionRegularised.interpolate(Interpolator)
    return DisplacementFunctionExperimental, DisplacementFunctionRegularised

def OutputDisplacementsXDMF(AnalysisName,ObjectMesh,DisplacementsExperimental,DisplacementsRegularised):
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, AnalysisName+"OutputExperimentalU.xdmf", "w") as xdmf_outfile:
        xdmf_outfile.write_mesh(ObjectMesh)
        DisplacementsExperimental.name = "Displacement"
        xdmf_outfile.write_function(DisplacementsExperimental)
        
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, AnalysisName+"OutputRegularisedU.xdmf", "w") as xdmf_outfile:
        xdmf_outfile.write_mesh(ObjectMesh)
        DisplacementsRegularised.name = "Displacement"
        xdmf_outfile.write_function(DisplacementsRegularised)
