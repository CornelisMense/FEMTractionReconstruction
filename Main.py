from Analysis import Analysis

analysis = Analysis("Config.yaml")

analysis.Preprocess()
analysis.VisualiseIntensities()
analysis.MeshObject()
analysis.CreateDisplacementField()
analysis.VisualiseDisplacements()
