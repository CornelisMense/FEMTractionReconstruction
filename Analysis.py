import yaml
import numpy as np

from DataPreprocessing import ReadTiff, SetWindow, Threshold, Smoothing, Downscale, CalculateIntensityDrift, CalculateMeshScale
from ScaffoldVisualiser import CreateBoxMesh, ReadBoxMeshXDMF, CreateIntensityInterpolator, InterpolateIntensity, ExportIntensityXDMF
from DisplacementCalculation import CalculateBSpline, ImportNumpyArray, ExportNumpyArray, CreateDisplacementInterpolator, InterpolateDisplacement, ExportDisplacementXDMF, CorrectDisplacementDrift
from MarchingCubesMesher import MCMesherSTL, ImportSTL, MesherXDMF, ImportMeshXDMF

class Analysis:
    class Displacements:
        def __init__(self, analysis):
            self.Analysis = analysis
            self.NumpyArray = None
            self.ExperimentalFEM = None
            self.RegularisedFEM = None
            self.Interpolator = None
            
        def BSplineCalculation(self):
            ImageUndeformed = self.Analysis.Images.Images["Undeformed"]
            ImageDeformed = self.Analysis.Images.Images["Deformed"]
            ResolutionZ = self.Analysis.cfg["ResolutionZ"]
            BSplineBasis = self.Analysis.cfg["BSplineBasis"]
            self.NumpyArray = CalculateBSpline(ImageUndeformed,ImageDeformed,ResolutionZ,BSplineBasis)

        def ImportNumpy(self):
            ArrayName = self.Analysis.cfg["DisplacementFieldImportName"]
            self.NumpyArray = ImportNumpyArray(ArrayName)

        def ExportNumpy(self):
            DisplacementsNumpy = self.NumpyArray
            ExportNumpyArray(DisplacementsNumpy)

        def CreateInterpolator(self):
            DisplacementsNumpy = self.NumpyArray
            MeshScale = self.Analysis.Images.MeshScale
            self.Interpolator = CreateDisplacementInterpolator(DisplacementsNumpy,MeshScale)

        def AutoCorrectDisplacementDrift(self):
            DisplacementsNumpy = self.NumpyArray
            BulkImage = self.Analysis.Images.Images["Bulk"]
            self.NumpyArray, self.Analysis.Results.DisplacementDrift = CorrectDisplacementDrift(DisplacementsNumpy,BulkImage)

        def Interpolate(self):
            Interpolator = self.Interpolator
            ObjectMesh = self.Analysis.Meshes.ObjectMesh
            self.ExperimentalFEM, self.RegularisedFEM = InterpolateDisplacement(Interpolator,ObjectMesh)

        def ExportXDMF(self):
            Name = self.Analysis.cfg["Name"]
            ObjectMesh = self.Analysis.Meshes.BoxMesh
            DisplacementsExperimental = self.ExperimentalFEM
            DisplacementsRegularised = self.RegularisedFEM
            ExportDisplacementsXDMF(Name,ObjectMesh,DisplacementsExperimental,DisplacementsRegularised)
            
    class Meshes:
        def __init__(self, analysis):
            self.Analysis = analysis
            self.BoxMesh = None
            self.ObjectMesh = None
            self.MeshName = None

        def BoxMesher(self):
            Name = self.Analysis.cfg["Name"]
            Resolution = self.Analysis.cfg["BoxMeshResolution"]
            MeshScale = self.Analysis.Images.MeshScale
            CreateBoxMesh(Name,Resolution,MeshScale)
            self.BoxMesh = ReadBoxMeshXDMF(Name)

        def MarchingCubesSTL(self):
            Name = self.Analysis.cfg["Name"]
            BulkField = self.Analysis.Images.Images["Bulk"]
            BulkThresholdValueMin = self.Analysis.cfg["ThresholdMinBulk"]
            self.MeshName = MCMesherSTL(Name,BulkField,BulkThresholdValueMin)

        def ImportSTL(self):
            self.MeshName = self.Analysis.cfg["STLImportName"]

        def STLtoXMDF(self):
            Name = self.MeshName
            MesherXDMF(Name)

        def ImportXDMF(self):
            Name = self.MeshName
            self.ObjectMesh = ImportMeshXDMF(MeshName)
        

    class Images:
        def __init__(self, analysis):
            self.Analysis = analysis
            self.Images = {}
            self.ImagesBinary = {}
            self.ImagesSmoothed = {}
            self.ImagesFEM = {}
            self.Interpolators = {}
            self.MeshScale = None

        def LoadTiff(self):
            imageUndeformed, imageDeformed, imageBulk, imageCells = ReadTiff(self.Analysis.cfg["ImagePaths"],self.Analysis.cfg["CellSegmentation"])
            self.Images = {"Undeformed": imageUndeformed,"Deformed": imageDeformed,"Bulk": imageBulk,}
            if imageCells is not None:
                self.Images["Cells"] = imageCells)
            MeshScaleImage = self.Images["Undeformed"]
            ResolutionZ = self.Analysis.Images.ResolutionZ
            self.MeshScale = CalculateMeshScale(MeshScaleImage,ResolutionZ)

        def WindowImages(self):
            MeshRange = np.array([self.Analysis.cfg["RangeX"],self.Analysis.cfg["RangeY"],self.Analysis.cfg["RangeZ"]])
            for name, image in self.Images.items():
                Window = np.array([[self.Analysis.cfg[f"MinX{name}"],self.Analysis.cfg[f"MinX{name}"+MeshRange[0]]],
                                   [self.Analysis.cfg[f"MinY{name}"],self.Analysis.cfg[f"MinY{name}"+MeshRange[1]]],
                                   [self.Analysis.cfg[f"MinZ{name}"],self.Analysis.cfg[f"MinZ{name}"+MeshRange[2]]]])
                Offset = self.Analysis.cfg[f"Offet{name}"]
                self.Images[name] = SetWindow(image,Window,Offset)
            MeshScaleImage = self.Images["Undeformed"]
            ResolutionZ = self.Analysis.Images.ResolutionZ
            self.MeshScale = CalculateMeshScale(MeshScaleImage,ResolutionZ)

        def DownScaleImages(self):
            for name, image in self.Images.items():
                DownScaleFactor = self.Analysis.cfg["DownScaleFactor"] 
                self.Images[name]= DownScale(image,DownScaleFactor)
            MeshScaleImage = self.Images["Undeformed"]
            ResolutionZ = self.Analysis.Images.ResolutionZ
            self.MeshScale = CalculateMeshScale(MeshScaleImage,ResolutionZ)
            
        def ThresholdImages(self):
            for name, image in self.Images.items():
                Tmin = self.Analysis.cfg[f"ThresholdMin{name}"]
                Tmax = self.Analysis.cfg[f"ThresholdMax{name}"]
                Type = self.Analysis.cfg["ThresholdType"]
                Axis = self.Analysis.cfg["PhotoBleachingAxis"]
                self.ImagesBinary[name] = Threshold(image,(Tmin, Tmax),Type,Axis)

        def SmoothImages(self):
            for name, image in self.Images.items():
                Sigma = self.Analysis.cfg[f"SmoothingSigma{name}"]
                self.ImagesSmoothed[name] = Smoothing(image,Sigma)

        def CleanImages(self):
            for name, image in self.Images.items():
                if (self.ImagesSmoothed[name] is not None) and (self.ImagesBinary[name] is not None):
                    Smooth = self.ImagesSmoothed[name]
                    Threshold = self.ImagesBinary[name]
                    self.Images[name] = Smooth*Threshold

        def CreateInterpolators(self):
            MeshScale = self.MeshScale
            Type = self.Analysis.cfg["IntensityInteprolationType"]
            for name, image in self.Images.items():
                self.Interpolators[name] = CreateIntensityInterpolator(image,MeshScale,Type)

        def Interpolate(self):
            BoxMesh = self.Analysis.Meshes.BoxMesh
            for name, interpolator in self.Interpolators.items():
                self.ImagesFEM[name] = InterpolateIntensity(interpolator,BoxMesh)

        def ExportXDMF(self):
            BoxMesh = self.Analysis.Meshes.BoxMesh
            AnalysisName = self.Analysis.cfg["Name"]
            for name, image in self.ImagesFEM.items():
                ExportIntensityXDMF(name,image,BoxMesh,AnalysisName)

        def AutoCorrectIntensityDrift(Self):
            self.Analysis.Results.IntensityDrift["Deformed"] = CalculateIntensityDrift(self.Images["Deformed"],self.Images["Undeformed"])
            self.Analysis.Results.IntensityDrift["Bulk"] = CalculateIntensityDrift(self.Images["Bulk"],self.Images["Undeformed"])

    class Results:
        def __init__(self, analysis):
            self.Analysis = analysis
            self.IntensityDrift = {}
            self.DisplacementDrift = None
            self.DirectTraction = None
            self.RegularisedTraction = None

        def CalculateTotalForce(self):
            pass

        
    def __init__(self, ConfigFile):
        with open(ConfigFile.yaml) as f:
            self.cfg = yaml.safe_load(f)
        self.Images = Analysis.Images(self)
        self.Displacements = Analysis.Displacements(self)
        self.Meshes = Analysis.Meshes(self)
        self.Results = Analysis.Results(self)

    def PreProcess(self):
        self.Images.LoadTiff()
        self.Images.WindowImages()
        self.Images.DownScaleImages()
        self.Images.ThresholdImages()
        self.Images.SmoothImages()
        self.Images.CleanImages()

    def VisualiseIntensities(self):
        self.Meshes.BoxMesher()
        self.Images.CreateInterpolators()
        self.Images.Interpolate()
        self.Images.ExportXDMF()

    def MeshObject(self):
        self.Meshes.MarchingCubesSTL()
        self.Meshes.STLToXDMF()
        self.Meshes.ImportXDMF()

    def CreateDisplacementField(self):
        self.Displacements.BSplineCalculation()
        self.Displacements.AutoCorrectDisplacementDrift()
        self.Displacements.ExportNumpy()

    def VisualiseDisplacements(self):
        self.Displacements.CreateInterpolator()
        self.Displacements.Interpolate()
        self.Displacements.ExportXDMF()
    
    def RegulariseTractions(self):
        pass
    
    def ExportTractionXDMF(self):
        pass

    
        

    
