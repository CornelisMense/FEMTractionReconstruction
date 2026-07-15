import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from skimage import measure
from skimage.measure import block_reduce

def ReadTiff(Paths,CellSegmentation=False):
    with tifffile.TiffFile(Paths[0]) as tif:
        imageDeformed = tif.asarray()
    with tifffile.TiffFile(Paths[1]) as tif:
        imageUndeformed = tif.asarray()
    with tifffile.TiffFile(Paths[2]) as tif:
        imageBulk = tif.asarray()
    imageDeformed, imageUndeformed, imageBulk = imageDeformed[[2,1,0],:,:], imageUndeformed[[2,1,0],:,:], imageBulk[[2,1,0],:,:]
    imageCells = None
    if CellSegmentation:
        with tifffile.TiffFile(Paths[3]) as tif:
            imageCells = tif.asarray()
        imageCells = imageCells[[2,1,0],:,:]

    return imageUndeformed, imageDeformed, imageBulk, imageCells

def SetWindow(Array,Parameters,Offset=[0,0,0]):
    XMin, XMax, YMin, YMax, ZMin, ZMax = Parameters[0,0], Parameters[0,1], Parameters[1,0], Parameters[1,1], Parameters[2,0], Parameters[2,1]
    XOffset, YOffset, ZOffset = Offset[0], Offset[1], Offset[2]
    WindowedArray = Array[XMin+XOffset:XMax+XOffset,YMin+YOffset:YMax+YOffset,ZMin+Zoffset:ZMax+ZOffset]
    return WindowedArray

def Threshold(Array,Range,Type=None,Axis=0):
    if Type == 'PhotoBleaching':
        ThresholdValueMin = Range[0]
        ThresholdValueMax = Range[1]
        ThresholdArray = ThresholdValueMax + (ThresholdValueMin-ThresholdValueMax)*np.arange(ArrayShape[Axis])/ArrayShape[Axis]
        ArrayShape = np.shape(Array)
        DimShape = [1] * Array.ndim
        DimShape[Axis] = ArrayShape[Axis]
        ThresholdValue = ThresholdArray.reshape(DimShape)
        BinaryArray = (Array>ThresholdValue)
    else:
        ThresholdValue = Range[0]
        BinaryArray = (Array>ThresholdValue)
    return BindaryArray

def Smoothing(Array,Sigma):
    SmoothedArray = gaussian_filter(Array, sigma=LeverSigma)
    return SmoothedArray

def CalculateIntensityDrift(Deformed,UnDeformed):
    DeformedX = np.tile(np.arange(np.shape(Deformed)[2]).reshape(1,1,np.shape(Deformed)[2]),(np.shape(Deformed)[0],np.shape(Deformed)[1],1))
    DeformedY = np.tile(np.arange(np.shape(Deformed)[1]).reshape(1,np.shape(Deformed)[1],1),(np.shape(Deformed)[0],1,np.shape(Deformed)[2]))
    DeformedZ = np.tile(np.arange(np.shape(Deformed)[0]).reshape(np.shape(Deformed)[0],1,1),(1,np.shape(Deformed)[1],np.shape(Deformed)[2]))
    DeformedCoM = np.array([np.sum(DeformedZ*Deformed),np.sum(DeformedY*Deformed),np.sum(DeformedX*Deformed)])/np.sum(Deformed)
    UnDeformedX = np.tile(np.arange(np.shape(UnDeformed)[2]).reshape(1,1,np.shape(UnDeformed)[2]),(np.shape(UnDeformed)[0],np.shape(UnDeformed)[1],1))
    UnDeformedY = np.tile(np.arange(np.shape(UnDeformed)[1]).reshape(1,np.shape(UnDeformed)[1],1),(np.shape(UnDeformed)[0],1,np.shape(UnDeformed)[2]))
    UnDeformedZ = np.tile(np.arange(np.shape(UnDeformed)[0]).reshape(np.shape(UnDeformed)[0],1,1),(1,np.shape(UnDeformed)[1],np.shape(UnDeformed)[2]))
    UnDeformedCoM = np.array([np.sum(UnDeformedZ*UnDeformed),np.sum(UnDeformedY*UnDeformed),np.sum(UnDeformedX*UnDeformed)])/np.sum(UnDeformed)
    DriftVector = DeformedCoM-UnDeformedCoM
    return DriftVector

def DownScale(Array,DownScaleFactor,Function=np.mean):
    DownScaledArray = measure.block_reduce(Array, block_size=(DownScaleFactor,DownScaleFactor,DownScaleFactor), func=Function)
    return DOwnScaledArray

def CalculateMeshScale(Image,ResolutionZ):
    ImageShape = np.shape(Image)
    ScaleX = np.arange(ImageShape[0])
    ScaleY = np.arange(ImageShape[1])
    ScaleY = np.arange(ImageShape[2])*ResolutionZ
    MeshScale = [ScaleX,ScaleY,ScaleZ]
    return MeshScale



    
