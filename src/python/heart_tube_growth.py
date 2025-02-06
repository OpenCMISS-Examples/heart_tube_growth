#!/usr/bin/env python

# This is an example script to illustrate growth in a heart tube using OpenCMISS calls.
#
# This script allows exploration of different growth options in a tube. Various parameters can be changed such as the
# growth model and parameters of growth, finite elasticity constitutive law, elasticity boundary conditions, etc. 
#

# Import python libraries
import sys, os
import math
import numpy as np
from scipy import linalg

# Import OpenCMISS
from opencmiss.opencmiss import OpenCMISS_Python as oc

# Parameters for the script

# Tube mesh

NUMBER_OF_CIRCUMFRENTIAL_ELEMENTS_PER_QUARTER = 3
NUMBER_OF_LENGTH_ELEMENTS = 6
NUMBER_OF_WALL_ELEMENTS = 2

# Geometry

LENGTH = 10.0
INNER_RADIUS = 1.25
OUTER_RADIUS = 2.00

# Growth model

UNIFORM_GROWTH_MODEL = 1
LIMITED_GROWTH_MODEL = 2

GROWTH_MODEL = UNIFORM_GROWTH_MODEL
#GROWTH_MODEL = LIMITED_GROWTH_MODEL

# Growth rates and parameters

FIBRE_ALPHA = 0.005
FIBRE_BETA = 1.0
FIBRE_GAMMA = 1.1

SHEET_ALPHA = 0.005
SHEET_BETA = 1.0
SHEET_GAMMA = 1.2

NORMAL_ALPHA = 0.005
NORMAL_BETA = 1.0
NORMAL_GAMMA = 1.05

FIBRE_SHEET_ALPHA = 0.0
FIBRE_SHEET_BETA = 1.0
FIBRE_SHEET_GAMMA = 1.0

FIBRE_NORMAL_ALPHA = 0.0
FIBRE_NORMAL_BETA = 1.0
FIBRE_NORMAL_GAMMA = 1.0

SHEET_NORMAL_ALPHA = 0.0
SHEET_NORMAL_BETA = 1.0
SHEET_NORMAL_GAMMA = 1.0

# Fibres

USE_FIBRES = True
FIBRE_ANGLE = 0.0
HETEROGENEOUS_FIBRES = False

# Constitutive laws and parameters

ST_VENANT_KIRCHOFF_CONSTITUTIVE_LAW = 1
NEO_HOOKEAN_CONSTITUTIVE_LAW = 2
MOONEY_RIVLIN_CONSTITUTIVE_LAW = 3

#CONSTITUTIVE_LAW = NEO_HOOKEAN_CONSTITUTIVE_LAW
CONSTITUTIVE_LAW = MOONEY_RIVLIN_CONSTITUTIVE_LAW

# Material parameters

STVK_LAMBDA = 1.0
STVK_MU = 1.0

NH_C1 = 100000.0

MR_C1 = 2.0
MR_C2 = 4.0

# Initial hydrostatic pressure

INITIAL_HYDROSTATIC_PRESSURE = -8.0

# Loads

MID_TUBE_FORCE = 0.1

# Fitting smoothing parameters

TAU = 0.1
KAPPA = 0.05

# Simulation "times"

START_TIME = 0.0
STOP_TIME = 50.0
DELTA_TIME = 1.0

# Should not need to change any of the parameters below

NUMBER_OF_CIRCUMFRENTIAL_ELEMENTS = 4*NUMBER_OF_CIRCUMFRENTIAL_ELEMENTS_PER_QUARTER
NUMBER_OF_LENGTH_NODES = NUMBER_OF_LENGTH_ELEMENTS+1
NUMBER_OF_CIRCUMFRENTIAL_NODES = NUMBER_OF_CIRCUMFRENTIAL_ELEMENTS
NUMBER_OF_WALL_NODES = NUMBER_OF_WALL_ELEMENTS+1
NUMBER_OF_NODES = NUMBER_OF_CIRCUMFRENTIAL_ELEMENTS*(NUMBER_OF_LENGTH_ELEMENTS+1)*(NUMBER_OF_WALL_ELEMENTS+1)
NUMBER_OF_ELEMENTS = NUMBER_OF_CIRCUMFRENTIAL_ELEMENTS*NUMBER_OF_LENGTH_ELEMENTS*NUMBER_OF_WALL_ELEMENTS

NUMBER_OF_DIMENSIONS = 3
NUMBER_OF_XI = NUMBER_OF_DIMENSIONS
NUMBER_OF_TENSOR_COMPONENTS = NUMBER_OF_DIMENSIONS*NUMBER_OF_DIMENSIONS
NUMBER_OF_VOIGT_COMPONENTS = oc.NumberOfVoigtComponentsGet(NUMBER_OF_DIMENSIONS)
VOIGT11_COMPONENT = oc.TensorComponentsToVoigtComponentGet(NUMBER_OF_DIMENSIONS,1,1)
VOIGT22_COMPONENT = oc.TensorComponentsToVoigtComponentGet(NUMBER_OF_DIMENSIONS,2,2)
VOIGT33_COMPONENT = oc.TensorComponentsToVoigtComponentGet(NUMBER_OF_DIMENSIONS,3,3)
VOIGT12_COMPONENT = oc.TensorComponentsToVoigtComponentGet(NUMBER_OF_DIMENSIONS,1,2)
VOIGT13_COMPONENT = oc.TensorComponentsToVoigtComponentGet(NUMBER_OF_DIMENSIONS,1,3)
VOIGT23_COMPONENT = oc.TensorComponentsToVoigtComponentGet(NUMBER_OF_DIMENSIONS,2,3)
NUMBER_OF_TENSOR_COMPONENTS = oc.NumberOfTensorTwoComponentsGet(NUMBER_OF_DIMENSIONS)
TENSOR11_COMPONENT = oc.TensorTwoComponentsToComponentGet(NUMBER_OF_DIMENSIONS,1,1)
TENSOR21_COMPONENT = oc.TensorTwoComponentsToComponentGet(NUMBER_OF_DIMENSIONS,2,1)
TENSOR31_COMPONENT = oc.TensorTwoComponentsToComponentGet(NUMBER_OF_DIMENSIONS,3,1)
TENSOR12_COMPONENT = oc.TensorTwoComponentsToComponentGet(NUMBER_OF_DIMENSIONS,1,2)
TENSOR22_COMPONENT = oc.TensorTwoComponentsToComponentGet(NUMBER_OF_DIMENSIONS,2,2)
TENSOR32_COMPONENT = oc.TensorTwoComponentsToComponentGet(NUMBER_OF_DIMENSIONS,3,2)
TENSOR13_COMPONENT = oc.TensorTwoComponentsToComponentGet(NUMBER_OF_DIMENSIONS,1,3)
TENSOR23_COMPONENT = oc.TensorTwoComponentsToComponentGet(NUMBER_OF_DIMENSIONS,2,3)
TENSOR33_COMPONENT = oc.TensorTwoComponentsToComponentGet(NUMBER_OF_DIMENSIONS,3,3)

NUMBER_OF_GAUSS_XI = 3

TRICUBIC_HERMITE_MESH_COMPONENT = 1
TRILINEAR_LAGRANGE_MESH_COMPONENT = 2

(CONTEXT_USER_NUMBER,
 COORDINATE_SYSTEM_USER_NUMBER,
 REGION_USER_NUMBER,
 TRI_CUBIC_HERMITE_BASIS_USER_NUMBER,
 TRI_LINEAR_LAGRANGE_BASIS_USER_NUMBER,
 MESH_USER_NUMBER,
 DECOMPOSITION_USER_NUMBER,
 DECOMPOSER_USER_NUMBER,
 GEOMETRIC_FIELD_USER_NUMBER,
 FIBRE_FIELD_USER_NUMBER,
 ELASTICITY_DEPENDENT_FIELD_USER_NUMBER,
 ELASTICITY_MATERIALS_FIELD_USER_NUMBER,
 ELASTICITY_DERIVED_FIELD_USER_NUMBER,
 ELASTICITY_EQUATIONS_SET_FIELD_USER_NUMBER,
 ELASTICITY_EQUATIONS_SET_USER_NUMBER,
 GROWTH_CELLML_USER_NUMBER,
 GROWTH_CELLML_MODELS_FIELD_USER_NUMBER,
 GROWTH_CELLML_STATE_FIELD_USER_NUMBER,
 GROWTH_CELLML_PARAMETERS_FIELD_USER_NUMBER,
 GROWTH_CELLML_INTERMEDIATE_FIELD_USER_NUMBER,
 ELASTICITY_PROBLEM_USER_NUMBER,
 STRESS_FIELD_USER_NUMBER,
 LAMBDA_FIELD_USER_NUMBER,
 FITTING_DEPENDENT_FIELD_USER_NUMBER,
 FITTING_INDEPENDENT_FIELD_USER_NUMBER,
 FITTING_MATERIALS_FIELD_USER_NUMBER,
 FITTING_EQUATIONS_SET_FIELD_USER_NUMBER,
 FITTING_EQUATIONS_SET_USER_NUMBER,
 FITTING_PROBLEM_USER_NUMBER
 ) = range(1,30)


#-----------------------------------------------------------------------------------------------------------
# CONTEXT AND WORLD REGION
#-----------------------------------------------------------------------------------------------------------

context = oc.Context()
context.Create(CONTEXT_USER_NUMBER)

worldRegion = oc.Region()
context.WorldRegionGet(worldRegion)

#-----------------------------------------------------------------------------------------------------------
# DIAGNOSTICS AND COMPUTATIONAL NODE INFORMATION
#-----------------------------------------------------------------------------------------------------------

oc.OutputSetOn("HeartTubeGrowth")

#oc.DiagnosticsSetOn(oc.DiagnosticTypes.IN,[1,2,3,4,5],"Diagnostics",["FiniteElasticity_DeformationGradientTensorCalculate"])

# Get the computational nodes information
computationEnvironment = oc.ComputationEnvironment()
context.ComputationEnvironmentGet(computationEnvironment)
numberOfComputationalNodes = computationEnvironment.NumberOfWorldNodesGet()
computationalNodeNumber = computationEnvironment.WorldNodeNumberGet()

worldWorkGroup = oc.WorkGroup()
computationEnvironment.WorldWorkGroupGet(worldWorkGroup)

#-----------------------------------------------------------------------------------------------------------
# COORDINATE SYSTEM
#-----------------------------------------------------------------------------------------------------------

# Create a 3D rectangular cartesian coordinate system
coordinateSystem = oc.CoordinateSystem()
coordinateSystem.CreateStart(COORDINATE_SYSTEM_USER_NUMBER,context)
# Set the number of dimensions to 3
coordinateSystem.DimensionSet(NUMBER_OF_DIMENSIONS)
# Finish the creation of the coordinate system
coordinateSystem.CreateFinish()

#-----------------------------------------------------------------------------------------------------------
# REGION
#-----------------------------------------------------------------------------------------------------------

# Create a region and assign the coordinate system to the region
region = oc.Region()
region.CreateStart(REGION_USER_NUMBER,worldRegion)
region.LabelSet("HeartTubeRegion")
# Set the regions coordinate system to the 3D RC coordinate system that we have created
region.CoordinateSystemSet(coordinateSystem)
# Finish the creation of the region
region.CreateFinish()

#-----------------------------------------------------------------------------------------------------------
# BASIS
#-----------------------------------------------------------------------------------------------------------

# Define basis
# Start the creation of a tricubic Hermite basis function
tricubicHermiteBasis = oc.Basis()
tricubicHermiteBasis.CreateStart(TRI_CUBIC_HERMITE_BASIS_USER_NUMBER,context)
tricubicHermiteBasis.TypeSet(oc.BasisTypes.LAGRANGE_HERMITE_TP)
tricubicHermiteBasis.NumberOfXiSet(NUMBER_OF_XI)
tricubicHermiteBasis.InterpolationXiSet([oc.BasisInterpolationSpecifications.CUBIC_HERMITE]*NUMBER_OF_XI)
tricubicHermiteBasis.QuadratureNumberOfGaussXiSet([NUMBER_OF_GAUSS_XI]*NUMBER_OF_XI)
tricubicHermiteBasis.CreateFinish()
# Start the creation of a trilinear Hermite basis function
trilinearLagrangeBasis = oc.Basis()
trilinearLagrangeBasis.CreateStart(TRI_LINEAR_LAGRANGE_BASIS_USER_NUMBER,context)
trilinearLagrangeBasis.TypeSet(oc.BasisTypes.LAGRANGE_HERMITE_TP)
trilinearLagrangeBasis.NumberOfXiSet(NUMBER_OF_XI)
trilinearLagrangeBasis.InterpolationXiSet([oc.BasisInterpolationSpecifications.LINEAR_LAGRANGE]*NUMBER_OF_XI)
trilinearLagrangeBasis.QuadratureNumberOfGaussXiSet([NUMBER_OF_GAUSS_XI]*NUMBER_OF_XI)
trilinearLagrangeBasis.CreateFinish()

#-----------------------------------------------------------------------------------------------------------
# NODES
#-----------------------------------------------------------------------------------------------------------

# Define nodes for the mesh
nodes = oc.Nodes()
nodes.CreateStart(region,NUMBER_OF_NODES)
nodes.CreateFinish()

#-----------------------------------------------------------------------------------------------------------
# MESH
#-----------------------------------------------------------------------------------------------------------

mesh = oc.Mesh()

# Create the mesh. The mesh will have two components - 1. tricubic Hermite elements; 2. trilinear Lagrange elements
mesh.CreateStart(MESH_USER_NUMBER,region,NUMBER_OF_XI)
mesh.NumberOfComponentsSet(2)
mesh.NumberOfElementsSet(NUMBER_OF_ELEMENTS)

tricubicHermiteElements = oc.MeshElements()
tricubicHermiteElements.CreateStart(mesh,TRICUBIC_HERMITE_MESH_COMPONENT,tricubicHermiteBasis)
trilinearLagrangeElements = oc.MeshElements()
trilinearLagrangeElements.CreateStart(mesh,TRILINEAR_LAGRANGE_MESH_COMPONENT,trilinearLagrangeBasis)

elementNumber = 0
for wallElementIdx in range(1,NUMBER_OF_WALL_ELEMENTS+1):
    for lengthElementIdx in range(1,NUMBER_OF_LENGTH_ELEMENTS+1):
        for circumfrentialElementIdx in range(1,NUMBER_OF_CIRCUMFRENTIAL_ELEMENTS+1):
            elementNumber = elementNumber + 1
            localNode1 = circumfrentialElementIdx + (lengthElementIdx-1)*NUMBER_OF_CIRCUMFRENTIAL_NODES + \
                (wallElementIdx-1)*NUMBER_OF_CIRCUMFRENTIAL_NODES*NUMBER_OF_LENGTH_NODES
            if circumfrentialElementIdx == NUMBER_OF_CIRCUMFRENTIAL_ELEMENTS:
                localNode2 = 1 + (lengthElementIdx-1)*NUMBER_OF_CIRCUMFRENTIAL_NODES + \
                    (wallElementIdx-1)*NUMBER_OF_CIRCUMFRENTIAL_NODES*NUMBER_OF_LENGTH_NODES
            else:
                localNode2 = localNode1 + 1
            localNode3 = localNode1 + NUMBER_OF_CIRCUMFRENTIAL_NODES
            localNode4 = localNode2 + NUMBER_OF_CIRCUMFRENTIAL_NODES
            localNode5 = localNode1 + NUMBER_OF_CIRCUMFRENTIAL_NODES*NUMBER_OF_LENGTH_NODES
            localNode6 = localNode2 + NUMBER_OF_CIRCUMFRENTIAL_NODES*NUMBER_OF_LENGTH_NODES
            localNode7 = localNode3 + NUMBER_OF_CIRCUMFRENTIAL_NODES*NUMBER_OF_LENGTH_NODES
            localNode8 = localNode4 + NUMBER_OF_CIRCUMFRENTIAL_NODES*NUMBER_OF_LENGTH_NODES
            localNodes = [localNode1,localNode2,localNode3,localNode4,localNode5,localNode6,localNode7,localNode8]
            tricubicHermiteElements.NodesSet(elementNumber,localNodes)
            trilinearLagrangeElements.NodesSet(elementNumber,localNodes)

tricubicHermiteElements.CreateFinish()
trilinearLagrangeElements.CreateFinish()

# Finish the mesh creation
mesh.CreateFinish() 

#-----------------------------------------------------------------------------------------------------------
# MESH DECOMPOSITION
#-----------------------------------------------------------------------------------------------------------

# Create a decomposition for the mesh
decomposition = oc.Decomposition()
decomposition.CreateStart(DECOMPOSITION_USER_NUMBER,mesh)
# Set the decomposition to be a general decomposition with the specified number of domains
decomposition.TypeSet(oc.DecompositionTypes.CALCULATED)
# Finish the decomposition
decomposition.CreateFinish()

#-----------------------------------------------------------------------------------------------------------
# DECOMPOSER
#-----------------------------------------------------------------------------------------------------------

decomposer = oc.Decomposer()
decomposer.CreateStart(DECOMPOSER_USER_NUMBER,worldRegion,worldWorkGroup)
decompositionIndex = decomposer.DecompositionAdd(decomposition)
decomposer.CreateFinish()

#-----------------------------------------------------------------------------------------------------------
# GEOMETRIC FIELD
#-----------------------------------------------------------------------------------------------------------

# Create a field for the geometry
geometricField = oc.Field()
geometricField.CreateStart(GEOMETRIC_FIELD_USER_NUMBER,region)
# Set the decomposition to use
geometricField.DecompositionSet(decomposition)
geometricField.TypeSet(oc.FieldTypes.GEOMETRIC)
# Set the field label
geometricField.VariableLabelSet(oc.FieldVariableTypes.U,"Geometry")
# Set the domain to be used by the field components to be tricubic Hermite
for componentIdx in range(1,NUMBER_OF_DIMENSIONS+1):
    geometricField.ComponentMeshComponentSet(oc.FieldVariableTypes.U,componentIdx,TRICUBIC_HERMITE_MESH_COMPONENT)
# Set the scaling type
geometricField.ScalingTypeSet(oc.FieldScalingTypes.ARITHMETIC_MEAN)
# Finish creating the field
geometricField.CreateFinish()

# Create the geometric field
for wallNodeIdx in range(1,NUMBER_OF_WALL_NODES+1):
    for lengthNodeIdx in range(1,NUMBER_OF_LENGTH_NODES+1):
        for circumfrentialNodeIdx in range(1,NUMBER_OF_CIRCUMFRENTIAL_NODES+1):
            nodeNumber = circumfrentialNodeIdx + (lengthNodeIdx-1)*NUMBER_OF_CIRCUMFRENTIAL_NODES + \
                (wallNodeIdx-1)*NUMBER_OF_CIRCUMFRENTIAL_NODES*NUMBER_OF_LENGTH_NODES
            nodeDomain = decomposition.NodeDomainGet(TRICUBIC_HERMITE_MESH_COMPONENT,nodeNumber)
            if nodeDomain == computationalNodeNumber:
                radius = INNER_RADIUS + (OUTER_RADIUS - INNER_RADIUS)*float(wallNodeIdx-1)/float(NUMBER_OF_WALL_NODES)
                theta = float(circumfrentialNodeIdx-1)/float(NUMBER_OF_CIRCUMFRENTIAL_NODES)*2.0*math.pi
                x = radius*math.cos(theta)
                y = radius*math.sin(theta)
                xtangent = -math.sin(theta)
                ytangent = math.cos(theta)
                xnormal = math.cos(theta)
                ynormal = math.sin(theta)
                z = float(lengthNodeIdx-1)/float(NUMBER_OF_LENGTH_NODES)*LENGTH
                geometricField.ParameterSetUpdateNodeDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                        1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,1,x)
                geometricField.ParameterSetUpdateNodeDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                        1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,2,y)
                geometricField.ParameterSetUpdateNodeDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                        1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,3,z)
                geometricField.ParameterSetUpdateNodeDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                        1,oc.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber,1,xtangent)
                geometricField.ParameterSetUpdateNodeDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                        1,oc.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber,2,ytangent)
                geometricField.ParameterSetUpdateNodeDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                        1,oc.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber,3,0.0)
                geometricField.ParameterSetUpdateNodeDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                        1,oc.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber,1,0.0)
                geometricField.ParameterSetUpdateNodeDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                        1,oc.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber,2,0.0)
                geometricField.ParameterSetUpdateNodeDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                        1,oc.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber,3,1.0)
                geometricField.ParameterSetUpdateNodeDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                        1,oc.GlobalDerivativeConstants.GLOBAL_DERIV_S3,nodeNumber,1,xnormal)
                geometricField.ParameterSetUpdateNodeDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                        1,oc.GlobalDerivativeConstants.GLOBAL_DERIV_S3,nodeNumber,2,ynormal)
                geometricField.ParameterSetUpdateNodeDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                        1,oc.GlobalDerivativeConstants.GLOBAL_DERIV_S3,nodeNumber,3,0.0)

# Update the geometric field
geometricField.ParameterSetUpdateStart(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES)
geometricField.ParameterSetUpdateFinish(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES)

#-----------------------------------------------------------------------------------------------------------
# FIBRE FIELD
#-----------------------------------------------------------------------------------------------------------

if USE_FIBRES:
    # Create a fibre field and attach it to the geometric field
    fibreField = oc.Field()
    fibreField.CreateStart(FIBRE_FIELD_USER_NUMBER,region)
    fibreField.TypeSet(oc.FieldTypes.FIBRE)
    # Set the decomposition 
    fibreField.DecompositionSet(decomposition)
    # Set the geometric field
    fibreField.GeometricFieldSet(geometricField)
    # Set the field variable label
    fibreField.VariableLabelSet(oc.FieldVariableTypes.U,"Fibre")
    # Set the fibre field to use trilinear-Lagrange elements
    fibreField.NumberOfComponentsSet(oc.FieldVariableTypes.U,NUMBER_OF_XI)
    for componentIdx in range(1,NUMBER_OF_XI+1):
        fibreField.ComponentMeshComponentSet(oc.FieldVariableTypes.U,componentIdx,TRILINEAR_LAGRANGE_MESH_COMPONENT)
    # Finish creating the field
    fibreField.CreateFinish()
    #Initialise the fibre field
    for wallNodeIdx in range(1,NUMBER_OF_WALL_NODES+1):
        for lengthNodeIdx in range(1,NUMBER_OF_LENGTH_NODES+1):
            for circumfrentialNodeIdx in range(1,NUMBER_OF_CIRCUMFRENTIAL_NODES+1):
                nodeNumber = circumfrentialNodeIdx + (lengthNodeIdx-1)*NUMBER_OF_CIRCUMFRENTIAL_NODES + \
                    (wallNodeIdx-1)*NUMBER_OF_CIRCUMFRENTIAL_NODES*NUMBER_OF_LENGTH_NODES
                nodeDomain = decomposition.NodeDomainGet(TRILINEAR_LAGRANGE_MESH_COMPONENT,nodeNumber)
                if nodeDomain == computationalNodeNumber:
                    # Set the fibre angle
                    if HETEROGENEOUS_FIBRES == True:
                        theta = float(circumfrentialNodeIdx-1)/float(NUMBER_OF_CIRCUMFRENTIAL_NODES)*2.0*math.pi
                        angle = FIBRE_ANGLE*math.sin(theta)
                    else:
                        angle = FIBRE_ANGLE
                    fibreField.ParameterSetUpdateNodeDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                        1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,1,angle)
                    fibreField.ParameterSetUpdateNodeDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                        1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,2,0.0)
                    fibreField.ParameterSetUpdateNodeDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,
                                                        1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,3,0.0)
    # Update the fibre field
    fibreField.ParameterSetUpdateStart(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES)
    fibreField.ParameterSetUpdateFinish(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES)

#-----------------------------------------------------------------------------------------------------------
# ELASTICITY EQUATION SETS
#-----------------------------------------------------------------------------------------------------------

# Create the equations_set
elasticityEquationsSetField = oc.Field()
elasticityEquationsSet = oc.EquationsSet()
# Specify a finite elasticity equations set with the growth and constitutive law in CellML
if CONSTITUTIVE_LAW == ST_VENANT_KIRCHOFF_CONSTITUTIVE_LAW:
    elasticityEquationsSetSpecification = [oc.EquationsSetClasses.ELASTICITY,
                                           oc.EquationsSetTypes.FINITE_ELASTICITY,
                                           oc.EquationsSetSubtypes.FULL_TENSOR_GROWTH_LAW_ST_VENANT]
elif CONSTITUTIVE_LAW == NEO_HOOKEAN_CONSTITUTIVE_LAW:
    elasticityEquationsSetSpecification = [oc.EquationsSetClasses.ELASTICITY,
                                           oc.EquationsSetTypes.FINITE_ELASTICITY,
                                           oc.EquationsSetSubtypes.FULL_TENSOR_GROWTH_LAW_NEO_HOOKEAN]
elif CONSTITUTIVE_LAW == MOONEY_RIVLIN_CONSTITUTIVE_LAW:
    elasticityEquationsSetSpecification = [oc.EquationsSetClasses.ELASTICITY,
                                           oc.EquationsSetTypes.FINITE_ELASTICITY,
                                           oc.EquationsSetSubtypes.FULL_TENSOR_GROWTH_LAW_MOONEY_RIVLIN]
else:
    sys.exit('ERROR: Unknown constitutive law.')
        
if USE_FIBRES:
    elasticityEquationsSet.CreateStart(ELASTICITY_EQUATIONS_SET_USER_NUMBER,region,fibreField,
                                       elasticityEquationsSetSpecification,ELASTICITY_EQUATIONS_SET_FIELD_USER_NUMBER,
                                       elasticityEquationsSetField)
else:
    elasticityEquationsSet.CreateStart(ELASTICITY_EQUATIONS_SET_USER_NUMBER,region,geometricField,
                                       elasticityEquationsSetSpecification,ELASTICITY_EQUATIONS_SET_FIELD_USER_NUMBER,
                                       elasticityEquationsSetField)
elasticityEquationsSet.CreateFinish()

#-----------------------------------------------------------------------------------------------------------
# ELASTITICY DEPENDENT FIELD
#-----------------------------------------------------------------------------------------------------------

# Create the dependent field
elasticityDependentField = oc.Field()
elasticityDependentField.CreateStart(ELASTICITY_DEPENDENT_FIELD_USER_NUMBER,region)
elasticityDependentField.TypeSet(oc.FieldTypes.GEOMETRIC_GENERAL)
# Set the decomposition
elasticityDependentField.DecompositionSet(decomposition)
# Set the geometric field
elasticityDependentField.GeometricFieldSet(geometricField) 
elasticityDependentField.DependentTypeSet(oc.FieldDependentTypes.DEPENDENT)
# Set the field variables for displacement, traction, and growth
elasticityDependentField.NumberOfVariablesSet(3)
elasticityDependentField.VariableTypesSet([oc.FieldVariableTypes.U,
                                           oc.FieldVariableTypes.T,
                                           oc.FieldVariableTypes.U3])
elasticityDependentField.VariableLabelSet(oc.FieldVariableTypes.U,"Dependent")
elasticityDependentField.VariableLabelSet(oc.FieldVariableTypes.T,"Traction")
elasticityDependentField.VariableLabelSet(oc.FieldVariableTypes.U3,"GrowthFactors")
elasticityDependentField.NumberOfComponentsSet(oc.FieldVariableTypes.U,NUMBER_OF_DIMENSIONS+1)
elasticityDependentField.NumberOfComponentsSet(oc.FieldVariableTypes.T,NUMBER_OF_DIMENSIONS+1)
elasticityDependentField.NumberOfComponentsSet(oc.FieldVariableTypes.U3,NUMBER_OF_VOIGT_COMPONENTS)
for componentIdx in range(1,NUMBER_OF_DIMENSIONS+1):
    # Set the displacments to use tri-cubic Hermite elements
    elasticityDependentField.ComponentMeshComponentSet(oc.FieldVariableTypes.U,componentIdx,
                                                       TRICUBIC_HERMITE_MESH_COMPONENT)
    # Set the tractions to use tri-cubic Hermite elements
    elasticityDependentField.ComponentMeshComponentSet(oc.FieldVariableTypes.T,componentIdx,
                                                       TRICUBIC_HERMITE_MESH_COMPONENT)
# Set the hydrostatic pressure to use tri-linear Lagrange elements
elasticityDependentField.ComponentMeshComponentSet(oc.FieldVariableTypes.U,NUMBER_OF_DIMENSIONS+1,
                                                   TRILINEAR_LAGRANGE_MESH_COMPONENT)
elasticityDependentField.ComponentMeshComponentSet(oc.FieldVariableTypes.T,NUMBER_OF_DIMENSIONS+1,
                                                   TRILINEAR_LAGRANGE_MESH_COMPONENT)
for componentIdx in range(1,NUMBER_OF_VOIGT_COMPONENTS+1):
    # Set the growth factors to use tri-linear Lagrange elements
    elasticityDependentField.ComponentMeshComponentSet(oc.FieldVariableTypes.U3,componentIdx,
                                                       TRILINEAR_LAGRANGE_MESH_COMPONENT)
    # Set the growth factors to be Gauss point based.
    elasticityDependentField.ComponentInterpolationSet(oc.FieldVariableTypes.U3,componentIdx,
                                                       oc.FieldInterpolationTypes.GAUSS_POINT_BASED)
# Set the field scaling
elasticityDependentField.fieldScalingType = oc.FieldScalingTypes.ARITHMETIC_MEAN
# Finish creating the field
elasticityDependentField.CreateFinish()

# Initialise dependent field from undeformed geometry
for componentIdx in range(1,NUMBER_OF_DIMENSIONS+1):
    oc.Field.ParametersToFieldParametersComponentCopy(
        geometricField,oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,componentIdx,
        elasticityDependentField,oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,componentIdx)
# Initialise the hydrostatic pressure
oc.Field.ComponentValuesInitialiseDP(elasticityDependentField,oc.FieldVariableTypes.U,
                                     oc.FieldParameterSetTypes.VALUES,NUMBER_OF_DIMENSIONS+1,
                                     INITIAL_HYDROSTATIC_PRESSURE)

# Set up the equation set dependent field
elasticityEquationsSet.DependentCreateStart(ELASTICITY_DEPENDENT_FIELD_USER_NUMBER,elasticityDependentField)
elasticityEquationsSet.DependentCreateFinish()

#-----------------------------------------------------------------------------------------------------------
# ELASTICITY MATERIALS FIELD
#-----------------------------------------------------------------------------------------------------------

# Set up the equation set materials field
elasticityMaterialsField = oc.Field()
elasticityEquationsSet.MaterialsCreateStart(ELASTICITY_MATERIALS_FIELD_USER_NUMBER,elasticityMaterialsField)
elasticityEquationsSet.MaterialsCreateFinish()

# Specify the material constitutive law parameters
if CONSTITUTIVE_LAW == ST_VENANT_KIRCHOFF_CONSTITUTIVE_LAW:
    oc.Field.ComponentValuesInitialiseDP(elasticityMaterialsField,oc.FieldVariableTypes.U,
                                         oc.FieldParameterSetTypes.VALUES,1,STVK_LAMBDA)
    oc.Field.ComponentValuesInitialiseDP(elasticityMaterialsField,oc.FieldVariableTypes.U,
                                         oc.FieldParameterSetTypes.VALUES,2,STVK_MU)
elif CONSTITUTIVE_LAW == NEO_HOOKEAN_CONSTITUTIVE_LAW:
    oc.Field.ComponentValuesInitialiseDP(elasticityMaterialsField,oc.FieldVariableTypes.U,
                                         oc.FieldParameterSetTypes.VALUES,1,NH_C1)
elif CONSTITUTIVE_LAW == MOONEY_RIVLIN_CONSTITUTIVE_LAW:
    oc.Field.ComponentValuesInitialiseDP(elasticityMaterialsField,oc.FieldVariableTypes.U,
                                         oc.FieldParameterSetTypes.VALUES,1,MR_C1)
    oc.Field.ComponentValuesInitialiseDP(elasticityMaterialsField,oc.FieldVariableTypes.U,
                                         oc.FieldParameterSetTypes.VALUES,2,MR_C2)
else:
    sys.exit('ERROR: Unknown constitutive law.')
        
#-----------------------------------------------------------------------------------------------------------
# ELASTICITY DERIVED FIELD
#-----------------------------------------------------------------------------------------------------------

# Create the derived field
elasticityDerivedField = oc.Field()
elasticityDerivedField.CreateStart(ELASTICITY_DERIVED_FIELD_USER_NUMBER,region)
elasticityDerivedField.TypeSet(oc.FieldTypes.GENERAL)
# Set the decomposition
elasticityDerivedField.DecompositionSet(decomposition)
# Set the geometric field
elasticityDerivedField.GeometricFieldSet(geometricField) 
elasticityDerivedField.DependentTypeSet(oc.FieldDependentTypes.DEPENDENT)
# Set the field variables for displacement, traction, and growth
elasticityDerivedField.NumberOfVariablesSet(5)
elasticityDerivedField.VariableTypesSet([oc.FieldVariableTypes.U1,
                                         oc.FieldVariableTypes.U2,
                                         oc.FieldVariableTypes.U3,
                                         oc.FieldVariableTypes.U4,
                                         oc.FieldVariableTypes.U5])
elasticityDerivedField.VariableLabelSet(oc.FieldVariableTypes.U1,"DeformationGradient")
elasticityDerivedField.VariableLabelSet(oc.FieldVariableTypes.U2,"DeformationGradientFibre")
elasticityDerivedField.VariableLabelSet(oc.FieldVariableTypes.U3,"RightCauchyGreen")
elasticityDerivedField.VariableLabelSet(oc.FieldVariableTypes.U4,"CauchyStress")
elasticityDerivedField.VariableLabelSet(oc.FieldVariableTypes.U5,"CauchyFibreStress")
elasticityDerivedField.NumberOfComponentsSet(oc.FieldVariableTypes.U1,NUMBER_OF_TENSOR_COMPONENTS)
elasticityDerivedField.NumberOfComponentsSet(oc.FieldVariableTypes.U2,NUMBER_OF_TENSOR_COMPONENTS)
elasticityDerivedField.NumberOfComponentsSet(oc.FieldVariableTypes.U3,NUMBER_OF_VOIGT_COMPONENTS)
elasticityDerivedField.NumberOfComponentsSet(oc.FieldVariableTypes.U4,NUMBER_OF_VOIGT_COMPONENTS)
elasticityDerivedField.NumberOfComponentsSet(oc.FieldVariableTypes.U5,NUMBER_OF_VOIGT_COMPONENTS)
for componentIdx in range(1,NUMBER_OF_TENSOR_COMPONENTS+1):
    # Set the components to use tricubic Hermite Gauss point based mesh
    elasticityDerivedField.ComponentMeshComponentSet(oc.FieldVariableTypes.U1,componentIdx,
                                                     TRICUBIC_HERMITE_MESH_COMPONENT)
    elasticityDerivedField.ComponentInterpolationSet(oc.FieldVariableTypes.U1,componentIdx,
                                                     oc.FieldInterpolationTypes.GAUSS_POINT_BASED)
    elasticityDerivedField.ComponentMeshComponentSet(oc.FieldVariableTypes.U2,componentIdx,
                                                     TRICUBIC_HERMITE_MESH_COMPONENT)
    elasticityDerivedField.ComponentInterpolationSet(oc.FieldVariableTypes.U2,componentIdx,
                                                     oc.FieldInterpolationTypes.GAUSS_POINT_BASED)
for componentIdx in range(1,NUMBER_OF_VOIGT_COMPONENTS+1):
    # Set the components to use tricubic Hermite Gauss point based mesh
    elasticityDerivedField.ComponentMeshComponentSet(oc.FieldVariableTypes.U3,componentIdx,
                                                     TRICUBIC_HERMITE_MESH_COMPONENT)
    elasticityDerivedField.ComponentInterpolationSet(oc.FieldVariableTypes.U3,componentIdx,
                                                     oc.FieldInterpolationTypes.GAUSS_POINT_BASED)
    elasticityDerivedField.ComponentMeshComponentSet(oc.FieldVariableTypes.U4,componentIdx,
                                                     TRICUBIC_HERMITE_MESH_COMPONENT)
    elasticityDerivedField.ComponentInterpolationSet(oc.FieldVariableTypes.U4,componentIdx,
                                                     oc.FieldInterpolationTypes.GAUSS_POINT_BASED)
    elasticityDerivedField.ComponentMeshComponentSet(oc.FieldVariableTypes.U5,componentIdx,
                                                     TRICUBIC_HERMITE_MESH_COMPONENT)
    elasticityDerivedField.ComponentInterpolationSet(oc.FieldVariableTypes.U5,componentIdx,
                                                     oc.FieldInterpolationTypes.GAUSS_POINT_BASED)
# Set the field scaling
elasticityDerivedField.fieldScalingType = oc.FieldScalingTypes.ARITHMETIC_MEAN
# Finish creating the field
elasticityDerivedField.CreateFinish()

# Set the derived variable types
elasticityEquationsSet.DerivedCreateStart(ELASTICITY_DERIVED_FIELD_USER_NUMBER,elasticityDerivedField)
elasticityEquationsSet.DerivedVariableSet(oc.EquationsSetDerivedTensorTypes.DEFORMATION_GRADIENT,
                                          oc.FieldVariableTypes.U1)
elasticityEquationsSet.DerivedVariableSet(oc.EquationsSetDerivedTensorTypes.DEFORMATION_GRADIENT_FIBRE,
                                          oc.FieldVariableTypes.U2)
elasticityEquationsSet.DerivedVariableSet(oc.EquationsSetDerivedTensorTypes.R_CAUCHY_GREEN_DEFORMATION,
                                          oc.FieldVariableTypes.U3)
elasticityEquationsSet.DerivedVariableSet(oc.EquationsSetDerivedTensorTypes.CAUCHY_STRESS,
                                          oc.FieldVariableTypes.U4)
elasticityEquationsSet.DerivedVariableSet(oc.EquationsSetDerivedTensorTypes.CAUCHY_STRESS_FIBRE,
                                          oc.FieldVariableTypes.U5)
elasticityEquationsSet.DerivedCreateFinish()

#-----------------------------------------------------------------------------------------------------------
# ELASTICITY EQUATIONS
#-----------------------------------------------------------------------------------------------------------

# Create equations
elasticityEquations = oc.Equations()
elasticityEquationsSet.EquationsCreateStart(elasticityEquations)
# Use sparse equations
elasticityEquations.SparsityTypeSet(oc.EquationsSparsityTypes.SPARSE)
# Do not output any equations information
elasticityEquations.OutputTypeSet(oc.EquationsOutputTypes.NONE)
# Finish creating the equations
elasticityEquationsSet.EquationsCreateFinish()

#-----------------------------------------------------------------------------------------------------------
# CELLML GROWTH MODEL
#-----------------------------------------------------------------------------------------------------------

# Set up the growth CellML model
growthCellML = oc.CellML()
growthCellML.CreateStart(GROWTH_CELLML_USER_NUMBER,region)
if GROWTH_MODEL == UNIFORM_GROWTH_MODEL:
    # Create the CellML environment for the uniform growth law
    growthCellMLIdx = growthCellML.ModelImport("uniformgrowth.cellml")
    # Flag the CellML variables that OpenCMISS will supply
    growthCellML.VariableSetAsKnown(growthCellMLIdx,"main/fibrealpha")
    growthCellML.VariableSetAsKnown(growthCellMLIdx,"main/sheetalpha")
    growthCellML.VariableSetAsKnown(growthCellMLIdx,"main/normalalpha")
    growthCellML.VariableSetAsKnown(growthCellMLIdx,"main/fibresheetalpha")
    growthCellML.VariableSetAsKnown(growthCellMLIdx,"main/fibrenormalalpha")
    growthCellML.VariableSetAsKnown(growthCellMLIdx,"main/sheetnormalalpha")
elif GROWTH_MODEL == LIMITED_GROWTH_MODEL:
    # Create the CellML environment for the limited growth law
    growthCellMLIdx = growthCellML.ModelImport("limitedgrowth.cellml")
    # Flag the CellML variables that OpenCMISS will supply
    growthCellML.VariableSetAsKnown(growthCellMLIdx,"main/fibrealpha")
    growthCellML.VariableSetAsKnown(growthCellMLIdx,"main/fibrebeta")
    growthCellML.VariableSetAsKnown(growthCellMLIdx,"main/fibregamma")
    growthCellML.VariableSetAsKnown(growthCellMLIdx,"main/sheetalpha")
    growthCellML.VariableSetAsKnown(growthCellMLIdx,"main/sheetbeta")
    growthCellML.VariableSetAsKnown(growthCellMLIdx,"main/sheetgamma")
    growthCellML.VariableSetAsKnown(growthCellMLIdx,"main/normalalpha")
    growthCellML.VariableSetAsKnown(growthCellMLIdx,"main/normalbeta")
    growthCellML.VariableSetAsKnown(growthCellMLIdx,"main/normalgamma")
    growthCellML.VariableSetAsKnown(growthCellMLIdx,"main/fibresheetalpha")
    growthCellML.VariableSetAsKnown(growthCellMLIdx,"main/fibresheetbeta")
    growthCellML.VariableSetAsKnown(growthCellMLIdx,"main/fibresheetgamma")
    growthCellML.VariableSetAsKnown(growthCellMLIdx,"main/fibrenormalalpha")
    growthCellML.VariableSetAsKnown(growthCellMLIdx,"main/fibrenormalbeta")
    growthCellML.VariableSetAsKnown(growthCellMLIdx,"main/fibrenormalgamma")
    growthCellML.VariableSetAsKnown(growthCellMLIdx,"main/sheetnormalalpha")
    growthCellML.VariableSetAsKnown(growthCellMLIdx,"main/sheetnormalbeta")
    growthCellML.VariableSetAsKnown(growthCellMLIdx,"main/sheetnormalgamma")
else:
    sys.exit('ERROR: Unknown growth model.')
# Finish the growth CellML
growthCellML.CreateFinish()

#-----------------------------------------------------------------------------------------------------------
# CELLML GROWTH FIELD MAPS
#-----------------------------------------------------------------------------------------------------------

# Create CellML <--> OpenCMISS field maps
growthCellML.FieldMapsCreateStart()
# Field --> CellML maps
# CellML --> Field maps
growthCellML.CreateCellMLToFieldMap(growthCellMLIdx,"main/lambdaf",oc.FieldParameterSetTypes.VALUES,
	                            elasticityDependentField,oc.FieldVariableTypes.U3,VOIGT11_COMPONENT,
                                    oc.FieldParameterSetTypes.VALUES)
growthCellML.CreateCellMLToFieldMap(growthCellMLIdx,"main/lambdas",oc.FieldParameterSetTypes.VALUES,
	                            elasticityDependentField,oc.FieldVariableTypes.U3,VOIGT22_COMPONENT,
                                    oc.FieldParameterSetTypes.VALUES)
growthCellML.CreateCellMLToFieldMap(growthCellMLIdx,"main/lambdan",oc.FieldParameterSetTypes.VALUES,
                                    elasticityDependentField,oc.FieldVariableTypes.U3,VOIGT33_COMPONENT,
                                    oc.FieldParameterSetTypes.VALUES)
growthCellML.CreateCellMLToFieldMap(growthCellMLIdx,"main/lambdafs",oc.FieldParameterSetTypes.VALUES,
	                            elasticityDependentField,oc.FieldVariableTypes.U3,VOIGT12_COMPONENT,
                                    oc.FieldParameterSetTypes.VALUES)
growthCellML.CreateCellMLToFieldMap(growthCellMLIdx,"main/lambdafn",oc.FieldParameterSetTypes.VALUES,
	                            elasticityDependentField,oc.FieldVariableTypes.U3,VOIGT13_COMPONENT,
                                    oc.FieldParameterSetTypes.VALUES)
growthCellML.CreateCellMLToFieldMap(growthCellMLIdx,"main/lambdasn",oc.FieldParameterSetTypes.VALUES,
	                            elasticityDependentField,oc.FieldVariableTypes.U3,VOIGT23_COMPONENT,
                                    oc.FieldParameterSetTypes.VALUES)
growthCellML.FieldMapsCreateFinish()

#-----------------------------------------------------------------------------------------------------------
# CELLML GROWTH MODELS FIELD
#-----------------------------------------------------------------------------------------------------------

# Create the CELL models field
growthCellMLModelsField = oc.Field()
growthCellML.ModelsFieldCreateStart(GROWTH_CELLML_MODELS_FIELD_USER_NUMBER,growthCellMLModelsField)
growthCellMLModelsField.VariableLabelSet(oc.FieldVariableTypes.U,"GrowthModelMap")
growthCellML.ModelsFieldCreateFinish()

#-----------------------------------------------------------------------------------------------------------
# CELLML GROWTH STATE FIELD
#-----------------------------------------------------------------------------------------------------------

LAMBDA_F_COMPONENT = growthCellML.FieldComponentGet(growthCellMLIdx,oc.CellMLFieldTypes.STATE,"main/lambdaf")
LAMBDA_S_COMPONENT = growthCellML.FieldComponentGet(growthCellMLIdx,oc.CellMLFieldTypes.STATE,"main/lambdas")
LAMBDA_N_COMPONENT = growthCellML.FieldComponentGet(growthCellMLIdx,oc.CellMLFieldTypes.STATE,"main/lambdan")
LAMBDA_FS_COMPONENT = growthCellML.FieldComponentGet(growthCellMLIdx,oc.CellMLFieldTypes.STATE,"main/lambdafs")
LAMBDA_FN_COMPONENT = growthCellML.FieldComponentGet(growthCellMLIdx,oc.CellMLFieldTypes.STATE,"main/lambdafn")
LAMBDA_SN_COMPONENT = growthCellML.FieldComponentGet(growthCellMLIdx,oc.CellMLFieldTypes.STATE,"main/lambdasn")

# Create the CELL state field
growthCellMLStateField = oc.Field()
growthCellML.StateFieldCreateStart(GROWTH_CELLML_STATE_FIELD_USER_NUMBER,growthCellMLStateField)
growthCellMLStateField.VariableLabelSet(oc.FieldVariableTypes.U,"GrowthState")
growthCellML.StateFieldCreateFinish()

#-----------------------------------------------------------------------------------------------------------
# GROWTH CELLML PARAMETERS FIELD
#-----------------------------------------------------------------------------------------------------------

# Create the CELL parameters field
growthCellMLParametersField = oc.Field()
growthCellML.ParametersFieldCreateStart(GROWTH_CELLML_PARAMETERS_FIELD_USER_NUMBER,
                                        growthCellMLParametersField)
growthCellMLParametersField.VariableLabelSet(oc.FieldVariableTypes.U,"GrowthParameters")
growthCellML.ParametersFieldCreateFinish()

# Set the growth parameters fields
fibreAlphaComponent = growthCellML.FieldComponentGet(growthCellMLIdx,oc.CellMLFieldTypes.PARAMETERS,"main/fibrealpha")
sheetAlphaComponent = growthCellML.FieldComponentGet(growthCellMLIdx,oc.CellMLFieldTypes.PARAMETERS,"main/sheetalpha")
normalAlphaComponent = growthCellML.FieldComponentGet(growthCellMLIdx,oc.CellMLFieldTypes.PARAMETERS,"main/normalalpha")
fibreSheetAlphaComponent = growthCellML.FieldComponentGet(growthCellMLIdx,oc.CellMLFieldTypes.PARAMETERS,"main/fibresheetalpha")
fibreNormalAlphaComponent = growthCellML.FieldComponentGet(growthCellMLIdx,oc.CellMLFieldTypes.PARAMETERS,"main/fibrenormalalpha")
sheetNormalAlphaComponent = growthCellML.FieldComponentGet(growthCellMLIdx,oc.CellMLFieldTypes.PARAMETERS,"main/sheetnormalalpha")

# Initialise the growth rates (alpha's). Sheet and normal are heterogeneous and done below.
# Note: The use of the component values initialise method will set all the values to be the same. If you wish to
# vary the rates by individual Gauss point then use the method below for the sheet and normal values.

oc.Field.ComponentValuesInitialiseDP(growthCellMLParametersField,oc.FieldVariableTypes.U,
                                     oc.FieldParameterSetTypes.VALUES,fibreAlphaComponent,FIBRE_ALPHA)
oc.Field.ComponentValuesInitialiseDP(growthCellMLParametersField,oc.FieldVariableTypes.U,
                                     oc.FieldParameterSetTypes.VALUES,fibreSheetAlphaComponent,FIBRE_SHEET_ALPHA)
oc.Field.ComponentValuesInitialiseDP(growthCellMLParametersField,oc.FieldVariableTypes.U,
                                     oc.FieldParameterSetTypes.VALUES,fibreNormalAlphaComponent,FIBRE_NORMAL_ALPHA)
oc.Field.ComponentValuesInitialiseDP(growthCellMLParametersField,oc.FieldVariableTypes.U,
                                     oc.FieldParameterSetTypes.VALUES,sheetNormalAlphaComponent,SHEET_NORMAL_ALPHA)
# Allow for heterogeneity in sheet and normal growth rates (alpha's)
for wallElementIdx in range(1,NUMBER_OF_WALL_ELEMENTS+1):
    for lengthElementIdx in range(1,NUMBER_OF_LENGTH_ELEMENTS+1):
        for circumfrentialElementIdx in range(1,NUMBER_OF_CIRCUMFRENTIAL_ELEMENTS+1):
            elementNumber = circumfrentialElementIdx + (lengthElementIdx-1)*NUMBER_OF_CIRCUMFRENTIAL_ELEMENTS + \
                (wallElementIdx-1)*NUMBER_OF_CIRCUMFRENTIAL_ELEMENTS*NUMBER_OF_LENGTH_ELEMENTS
            elementDomain = decomposition.ElementDomainGet(elementNumber)
            if elementDomain == computationalNodeNumber:
                for xiIdx3 in range(1,NUMBER_OF_GAUSS_XI+1):
                    for xiIdx2 in range(1,NUMBER_OF_GAUSS_XI+1):
                        for xiIdx1 in range(1,NUMBER_OF_GAUSS_XI+1):
                            gaussPointNumber = xiIdx1 + (xiIdx2-1)*NUMBER_OF_GAUSS_XI + \
                                (xiIdx3-1)*NUMBER_OF_GAUSS_XI*NUMBER_OF_GAUSS_XI
                            radius = (float(wallElementIdx-1)+float(xiIdx3)/float(NUMBER_OF_GAUSS_XI+1))/ \
                                float(NUMBER_OF_WALL_ELEMENTS)
                            theta = (float(circumfrentialElementIdx-1)+float(xiIdx1)/float(NUMBER_OF_GAUSS_XI+1))/ \
                                float(NUMBER_OF_CIRCUMFRENTIAL_ELEMENTS)*2.0*math.pi    
                            length = (float(lengthElementIdx-1)+float(xiIdx2)/float(NUMBER_OF_GAUSS_XI+1))/ \
                                float(NUMBER_OF_LENGTH_ELEMENTS)
                            sheetAlpha = SHEET_ALPHA*(1.0 + math.cos(theta))
                            normalAlpha = NORMAL_ALPHA*(1.0 + radius)
                            growthCellMLParametersField.ParameterSetUpdateGaussPointDP(oc.FieldVariableTypes.U,
                                                                                       oc.FieldParameterSetTypes.VALUES,
                                                                                       gaussPointNumber,elementNumber,
                                                                                       sheetAlphaComponent,sheetAlpha)
                            growthCellMLParametersField.ParameterSetUpdateGaussPointDP(oc.FieldVariableTypes.U,
                                                                                       oc.FieldParameterSetTypes.VALUES,
                                                                                       gaussPointNumber,elementNumber,
                                                                                       normalAlphaComponent,normalAlpha)

if GROWTH_MODEL == LIMITED_GROWTH_MODEL:
    fibreBetaComponent = growthCellML.FieldComponentGet(growthCellMLIdx,oc.CellMLFieldTypes.PARAMETERS,"main/fibrebeta")
    fibreGammaComponent = growthCellML.FieldComponentGet(growthCellMLIdx,oc.CellMLFieldTypes.PARAMETERS,"main/fibregamma")
    sheetBetaComponent = growthCellML.FieldComponentGet(growthCellMLIdx,oc.CellMLFieldTypes.PARAMETERS,"main/sheetbeta")
    sheetGammaComponent = growthCellML.FieldComponentGet(growthCellMLIdx,oc.CellMLFieldTypes.PARAMETERS,"main/sheetgamma")
    normalBetaComponent = growthCellML.FieldComponentGet(growthCellMLIdx,oc.CellMLFieldTypes.PARAMETERS,"main/normalbeta")
    normalGammaComponent = growthCellML.FieldComponentGet(growthCellMLIdx,oc.CellMLFieldTypes.PARAMETERS,"main/normalgamma")
    fibreSheetBetaComponent = growthCellML.FieldComponentGet(growthCellMLIdx,oc.CellMLFieldTypes.PARAMETERS,"main/fibresheetbeta")
    fibreSheetGammaComponent = growthCellML.FieldComponentGet(growthCellMLIdx,oc.CellMLFieldTypes.PARAMETERS,"main/fibresheetgamma")
    fibreNormalBetaComponent = growthCellML.FieldComponentGet(growthCellMLIdx,oc.CellMLFieldTypes.PARAMETERS,"main/fibrenormalbeta")
    fibreNormalGammaComponent = growthCellML.FieldComponentGet(growthCellMLIdx,oc.CellMLFieldTypes.PARAMETERS,"main/fibrenormalgamma")
    sheetNormalBetaComponent = growthCellML.FieldComponentGet(growthCellMLIdx,oc.CellMLFieldTypes.PARAMETERS,"main/sheetnormalbeta")
    sheetNormalGammaComponent = growthCellML.FieldComponentGet(growthCellMLIdx,oc.CellMLFieldTypes.PARAMETERS,"main/sheetnormalgamma")
    # Initialise the beta and gamma parameters. The component value initialise will set all values to be the same. If you
    # wish for heterogeneity at the Gauss point level use the methods above for the sheet and normal alpha's.
    oc.Field.ComponentValuesInitialiseDP(growthCellMLParametersField,oc.FieldVariableTypes.U,
                                         oc.FieldParameterSetTypes.VALUES,fibreBetaComponent,FIBRE_BETA)
    oc.Field.ComponentValuesInitialiseDP(growthCellMLParametersField,oc.FieldVariableTypes.U,
                                         oc.FieldParameterSetTypes.VALUES,fibreGammaComponent,FIBRE_GAMMA)
    oc.Field.ComponentValuesInitialiseDP(growthCellMLParametersField,oc.FieldVariableTypes.U,
                                         oc.FieldParameterSetTypes.VALUES,sheetBetaComponent,SHEET_BETA)
    oc.Field.ComponentValuesInitialiseDP(growthCellMLParametersField,oc.FieldVariableTypes.U,
                                         oc.FieldParameterSetTypes.VALUES,sheetGammaComponent,SHEET_GAMMA)
    oc.Field.ComponentValuesInitialiseDP(growthCellMLParametersField,oc.FieldVariableTypes.U,
                                         oc.FieldParameterSetTypes.VALUES,normalBetaComponent,NORMAL_BETA)
    oc.Field.ComponentValuesInitialiseDP(growthCellMLParametersField,oc.FieldVariableTypes.U,
                                         oc.FieldParameterSetTypes.VALUES,normalGammaComponent,NORMAL_GAMMA)
    oc.Field.ComponentValuesInitialiseDP(growthCellMLParametersField,oc.FieldVariableTypes.U,
                                         oc.FieldParameterSetTypes.VALUES,fibreSheetBetaComponent,FIBRE_SHEET_BETA)
    oc.Field.ComponentValuesInitialiseDP(growthCellMLParametersField,oc.FieldVariableTypes.U,
                                         oc.FieldParameterSetTypes.VALUES,fibreSheetGammaComponent,FIBRE_SHEET_GAMMA)
    oc.Field.ComponentValuesInitialiseDP(growthCellMLParametersField,oc.FieldVariableTypes.U,
                                         oc.FieldParameterSetTypes.VALUES,fibreNormalBetaComponent,FIBRE_NORMAL_BETA)
    oc.Field.ComponentValuesInitialiseDP(growthCellMLParametersField,oc.FieldVariableTypes.U,
                                         oc.FieldParameterSetTypes.VALUES,fibreNormalGammaComponent,FIBRE_NORMAL_GAMMA)
    oc.Field.ComponentValuesInitialiseDP(growthCellMLParametersField,oc.FieldVariableTypes.U,
                                         oc.FieldParameterSetTypes.VALUES,sheetNormalBetaComponent,SHEET_NORMAL_BETA)
    oc.Field.ComponentValuesInitialiseDP(growthCellMLParametersField,oc.FieldVariableTypes.U,
                                         oc.FieldParameterSetTypes.VALUES,sheetNormalGammaComponent,SHEET_NORMAL_GAMMA)
    
growthCellMLParametersField.ParameterSetUpdateStart(oc.FieldVariableTypes.U,
                                                    oc.FieldParameterSetTypes.VALUES)
growthCellMLParametersField.ParameterSetUpdateFinish(oc.FieldVariableTypes.U,
                                                     oc.FieldParameterSetTypes.VALUES)
                            
#-----------------------------------------------------------------------------------------------------------
# GROWTH CELLML INTERMEDIATE FIELD
#-----------------------------------------------------------------------------------------------------------

# Create the CELL intermediate field
#growthCellMLIntermediateField = oc.Field()
#growthCellML.IntermediateFieldCreateStart(GROWTH_CELLML_INTERMEDIATE_FIELD_USER_NUMBER,
#                                          growthCellMLIntermediateField)
#growthCellMLIntermediateField.VariableLabelSet(oc.FieldVariableTypes.U,"GrowthIntermediate")
#growthCellML.IntermediateFieldCreateFinish()

#-----------------------------------------------------------------------------------------------------------
# ELASTICITY PROBLEM
#-----------------------------------------------------------------------------------------------------------

# Define the problem
elasticityProblem = oc.Problem()
elasticityProblemSpecification = [oc.ProblemClasses.ELASTICITY,
                                  oc.ProblemTypes.FINITE_ELASTICITY,
                                  oc.ProblemSubtypes.FINITE_ELASTICITY_WITH_GROWTH_CELLML]
elasticityProblem.CreateStart(ELASTICITY_PROBLEM_USER_NUMBER,context,elasticityProblemSpecification)
elasticityProblem.CreateFinish()

#-----------------------------------------------------------------------------------------------------------
# ELASTICITY CONTROL LOOPS
#-----------------------------------------------------------------------------------------------------------

# Create control loops
elasticityProblemLoop = oc.ControlLoop()
elasticityProblem.ControlLoopCreateStart()
elasticityProblem.ControlLoopGet([oc.ControlLoopIdentifiers.NODE],elasticityProblemLoop)
elasticityProblem.ControlLoopCreateFinish()

#-----------------------------------------------------------------------------------------------------------
# ELASTICITY SOLVERS
#-----------------------------------------------------------------------------------------------------------

# Create problem solvers
growthODEIntegrationSolver = oc.Solver()
elasticityNonlinearSolver = oc.Solver()
elasticityLinearSolver = oc.Solver()
elasticityProblem.SolversCreateStart()
elasticityProblem.SolverGet([oc.ControlLoopIdentifiers.NODE],1,growthODEIntegrationSolver)
elasticityProblem.SolverGet([oc.ControlLoopIdentifiers.NODE],2,elasticityNonlinearSolver)
elasticityNonlinearSolver.OutputTypeSet(oc.SolverOutputTypes.MONITOR)
elasticityNonlinearSolver.NewtonJacobianCalculationTypeSet(oc.JacobianCalculationTypes.EQUATIONS)
elasticityNonlinearSolver.NewtonLinearSolverGet(elasticityLinearSolver)
elasticityLinearSolver.LinearTypeSet(oc.LinearSolverTypes.DIRECT)
elasticityProblem.SolversCreateFinish()

#-----------------------------------------------------------------------------------------------------------
# ELASTICITY SOLVER EQUATIONS
#-----------------------------------------------------------------------------------------------------------

# Create elasticity equations and add equations set to solver equations
elasticityEquations = oc.SolverEquations()
elasticityProblem.SolverEquationsCreateStart()
elasticityNonlinearSolver.SolverEquationsGet(elasticityEquations)
elasticityEquations.SparsityTypeSet(oc.SolverEquationsSparsityTypes.SPARSE)
elasticityEquationsSetIndex = elasticityEquations.EquationsSetAdd(elasticityEquationsSet)
elasticityProblem.SolverEquationsCreateFinish()

#-----------------------------------------------------------------------------------------------------------
# GROWTH CELLML EQUATIONS
#-----------------------------------------------------------------------------------------------------------

# Create CellML equations and add growth and constitutive equations to the solvers
growthEquations = oc.CellMLEquations()
elasticityProblem.CellMLEquationsCreateStart()
growthODEIntegrationSolver.CellMLEquationsGet(growthEquations)
growthEquationsIndex = growthEquations.CellMLAdd(growthCellML)
elasticityProblem.CellMLEquationsCreateFinish()

#-----------------------------------------------------------------------------------------------------------
# ELASTICITY BOUNDARY CONDITIONS
#-----------------------------------------------------------------------------------------------------------

# Prescribe boundary conditions for the elasticity (with growth) problem.
elasticityBoundaryConditions = oc.BoundaryConditions()
elasticityEquations.BoundaryConditionsCreateStart(elasticityBoundaryConditions)

#Fix bottom ring
for wallNodeIdx in range(1,NUMBER_OF_WALL_NODES+1):
    for circumfrentialNodeIdx in range(1,NUMBER_OF_CIRCUMFRENTIAL_NODES+1):
        nodeNumber = circumfrentialNodeIdx + (wallNodeIdx-1)*NUMBER_OF_CIRCUMFRENTIAL_NODES*NUMBER_OF_LENGTH_NODES
        nodeDomain = decomposition.NodeDomainGet(TRICUBIC_HERMITE_MESH_COMPONENT,nodeNumber)
        if nodeDomain == computationalNodeNumber:
            # Fix z direction
            elasticityBoundaryConditions.AddNode(elasticityDependentField,oc.FieldVariableTypes.U,
                                                 1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,3,
                                                 oc.BoundaryConditionsTypes.FIXED,0.0)
            # Fix S1 (circumfrential) direction derivatives
            elasticityBoundaryConditions.AddNode(elasticityDependentField,oc.FieldVariableTypes.U,
                                                 1,oc.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber,1,
                                                 oc.BoundaryConditionsTypes.FIXED,0.0)
            elasticityBoundaryConditions.AddNode(elasticityDependentField,oc.FieldVariableTypes.U,
                                                 1,oc.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber,2,
                                                 oc.BoundaryConditionsTypes.FIXED,0.0)
            elasticityBoundaryConditions.AddNode(elasticityDependentField,oc.FieldVariableTypes.U,
                                                 1,oc.GlobalDerivativeConstants.GLOBAL_DERIV_S1,nodeNumber,3,
                                                 oc.BoundaryConditionsTypes.FIXED,0.0)
            # Fix S2 (length) direction derivatives
            elasticityBoundaryConditions.AddNode(elasticityDependentField,oc.FieldVariableTypes.U,
                                                 1,oc.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber,1,
                                                 oc.BoundaryConditionsTypes.FIXED,0.0)
            elasticityBoundaryConditions.AddNode(elasticityDependentField,oc.FieldVariableTypes.U,
                                                 1,oc.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber,2,
                                                 oc.BoundaryConditionsTypes.FIXED,0.0)
            elasticityBoundaryConditions.AddNode(elasticityDependentField,oc.FieldVariableTypes.U,
                                                 1,oc.GlobalDerivativeConstants.GLOBAL_DERIV_S2,nodeNumber,3,
                                                 oc.BoundaryConditionsTypes.FIXED,0.0)
            # Fix S3 (wall) direction derivatives
            elasticityBoundaryConditions.AddNode(elasticityDependentField,oc.FieldVariableTypes.U,
                                                 1,oc.GlobalDerivativeConstants.GLOBAL_DERIV_S3,nodeNumber,1,
                                                 oc.BoundaryConditionsTypes.FIXED,0.0)
            elasticityBoundaryConditions.AddNode(elasticityDependentField,oc.FieldVariableTypes.U,
                                                 1,oc.GlobalDerivativeConstants.GLOBAL_DERIV_S3,nodeNumber,2,
                                                 oc.BoundaryConditionsTypes.FIXED,0.0)
            elasticityBoundaryConditions.AddNode(elasticityDependentField,oc.FieldVariableTypes.U,
                                                 1,oc.GlobalDerivativeConstants.GLOBAL_DERIV_S3,nodeNumber,3,
                                             oc.BoundaryConditionsTypes.FIXED,0.0)
    #Set symmetry conditions on the ring to prevent rotation
    nodeNumber = 1 + (wallNodeIdx-1)*NUMBER_OF_CIRCUMFRENTIAL_NODES*NUMBER_OF_LENGTH_NODES
    nodeDomain = decomposition.NodeDomainGet(TRICUBIC_HERMITE_MESH_COMPONENT,nodeNumber)
    if nodeDomain == computationalNodeNumber:
        elasticityBoundaryConditions.AddNode(elasticityDependentField,oc.FieldVariableTypes.U,
                                             1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,2,
                                             oc.BoundaryConditionsTypes.FIXED,0.0)
    nodeNumber = nodeNumber + NUMBER_OF_CIRCUMFRENTIAL_ELEMENTS_PER_QUARTER
    nodeDomain = decomposition.NodeDomainGet(TRICUBIC_HERMITE_MESH_COMPONENT,nodeNumber)
    if nodeDomain == computationalNodeNumber:
        elasticityBoundaryConditions.AddNode(elasticityDependentField,oc.FieldVariableTypes.U,
                                             1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,1,
                                             oc.BoundaryConditionsTypes.FIXED,0.0)
    nodeNumber = nodeNumber + NUMBER_OF_CIRCUMFRENTIAL_ELEMENTS_PER_QUARTER
    nodeDomain = decomposition.NodeDomainGet(TRICUBIC_HERMITE_MESH_COMPONENT,nodeNumber)
    if nodeDomain == computationalNodeNumber:
        elasticityBoundaryConditions.AddNode(elasticityDependentField,oc.FieldVariableTypes.U,
                                             1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,2,
                                             oc.BoundaryConditionsTypes.FIXED,0.0)
    nodeNumber = nodeNumber + NUMBER_OF_CIRCUMFRENTIAL_ELEMENTS_PER_QUARTER
    nodeDomain = decomposition.NodeDomainGet(TRICUBIC_HERMITE_MESH_COMPONENT,nodeNumber)
    if nodeDomain == computationalNodeNumber:
        elasticityBoundaryConditions.AddNode(elasticityDependentField,oc.FieldVariableTypes.U,
                                             1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,1,
                                             oc.BoundaryConditionsTypes.FIXED,0.0)    
# Apply a normal force to the middle of the tube
nodeNumber = 1 + math.floor(NUMBER_OF_LENGTH_NODES/2)*NUMBER_OF_CIRCUMFRENTIAL_NODES + \
    (NUMBER_OF_WALL_NODES-1)*NUMBER_OF_CIRCUMFRENTIAL_NODES*NUMBER_OF_LENGTH_NODES
nodeDomain = decomposition.NodeDomainGet(TRICUBIC_HERMITE_MESH_COMPONENT,nodeNumber)
if nodeDomain == computationalNodeNumber:
    # Fix the normal to the tube (i.e., the S3 direction) at the node
    normalx = geometricField.ParameterSetGetNode(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES, 
                                                 1,oc.GlobalDerivativeConstants.GLOBAL_DERIV_S3,nodeNumber,1)
    normaly = geometricField.ParameterSetGetNode(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES, 
                                                 1,oc.GlobalDerivativeConstants.GLOBAL_DERIV_S3,nodeNumber,2)
    normalz = geometricField.ParameterSetGetNode(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES, 
                                                 1,oc.GlobalDerivativeConstants.GLOBAL_DERIV_S3,nodeNumber,3)
    # Set the normal force
    elasticityBoundaryConditions.SetNode(elasticityDependentField,oc.FieldVariableTypes.T,
                                         1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,1,
                                         oc.BoundaryConditionsTypes.FIXED,-normalx*MID_TUBE_FORCE)
    elasticityBoundaryConditions.SetNode(elasticityDependentField,oc.FieldVariableTypes.T,
                                         1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,2,
                                         oc.BoundaryConditionsTypes.FIXED,-normaly*MID_TUBE_FORCE)
    elasticityBoundaryConditions.SetNode(elasticityDependentField,oc.FieldVariableTypes.T,
                                         1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,3,
                                         oc.BoundaryConditionsTypes.FIXED,-normalz*MID_TUBE_FORCE)
        
elasticityEquations.BoundaryConditionsCreateFinish()

#-----------------------------------------------------------------------------------------------------------
# GROWTH STRESS FIELD
#-----------------------------------------------------------------------------------------------------------

# Create the stress field
stressField = oc.Field()
stressField.CreateStart(STRESS_FIELD_USER_NUMBER,region)
stressField.TypeSet(oc.FieldTypes.GENERAL)
# Set the decomposition
stressField.DecompositionSet(decomposition)
# Set the geometric field
stressField.GeometricFieldSet(geometricField)
stressField.ScalingTypeSet(oc.FieldScalingTypes.NONE)
# Set the field variables
stressField.NumberOfVariablesSet(1)
stressField.VariableTypesSet([oc.FieldVariableTypes.U])
# Set the variable label
stressField.VariableLabelSet(oc.FieldVariableTypes.U,"NodalStress")
# Set the components to be trilinear-Lagrange
stressField.NumberOfComponentsSet(oc.FieldVariableTypes.U,NUMBER_OF_VOIGT_COMPONENTS)
for componentIdx in range(1,NUMBER_OF_VOIGT_COMPONENTS+1):
    # Set the stress field component to be trilinear Lagrange
    stressField.ComponentMeshComponentSet(oc.FieldVariableTypes.U,componentIdx,TRILINEAR_LAGRANGE_MESH_COMPONENT)
    # Set the interpolation type to be nodal based
    stressField.ComponentInterpolationSet(oc.FieldVariableTypes.U,componentIdx,oc.FieldInterpolationTypes.NODE_BASED)
stressField.CreateFinish()

#-----------------------------------------------------------------------------------------------------------
# GROWTH LAMBDA FIELD
#-----------------------------------------------------------------------------------------------------------

# Create the lambda field
lambdaField = oc.Field()
lambdaField.CreateStart(LAMBDA_FIELD_USER_NUMBER,region)
lambdaField.TypeSet(oc.FieldTypes.GENERAL)
# Set the decomposition
lambdaField.DecompositionSet(decomposition)
# Set the geometric field
lambdaField.GeometricFieldSet(geometricField)
lambdaField.ScalingTypeSet(oc.FieldScalingTypes.NONE)
# Set the field variables
lambdaField.NumberOfVariablesSet(1)
lambdaField.VariableTypesSet([oc.FieldVariableTypes.U])
# Set the variable label
lambdaField.VariableLabelSet(oc.FieldVariableTypes.U,"NodalLambda")
lambdaField.NumberOfComponentsSet(oc.FieldVariableTypes.U,NUMBER_OF_VOIGT_COMPONENTS)
for componentIdx in range(1,NUMBER_OF_VOIGT_COMPONENTS+1):
    # Set the components to be trilinear-Lagrange
    lambdaField.ComponentMeshComponentSet(oc.FieldVariableTypes.U,componentIdx,TRILINEAR_LAGRANGE_MESH_COMPONENT)
    # Set the interpolation type to be nodal based
    lambdaField.ComponentInterpolationSet(oc.FieldVariableTypes.U,componentIdx,oc.FieldInterpolationTypes.NODE_BASED)
lambdaField.CreateFinish()

# Initialise the lambda field
for componentIdx in range(1,NUMBER_OF_DIMENSIONS+1):
    lambdaField.ComponentValuesInitialiseDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,componentIdx,1.0)
for componentIdx in range(NUMBER_OF_DIMENSIONS+1,NUMBER_OF_VOIGT_COMPONENTS+1):
    lambdaField.ComponentValuesInitialiseDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,componentIdx,0.0)

#-----------------------------------------------------------------------------------------------------------
# FITTING EQUATIONS SET
#-----------------------------------------------------------------------------------------------------------

# Create Gauss point fitting equations set
fittingEquationsSetSpecification = [oc.EquationsSetClasses.FITTING,
                             oc.EquationsSetTypes.GAUSS_FITTING_EQUATION,
                             oc.EquationsSetSubtypes.GENERALISED_GAUSS_FITTING,
                             oc.EquationsSetFittingSmoothingTypes.SOBOLEV_VALUE]
fittingEquationsSetField = oc.Field()
fittingEquationsSet = oc.EquationsSet()
fittingEquationsSet.CreateStart(FITTING_EQUATIONS_SET_USER_NUMBER,region,geometricField,
        fittingEquationsSetSpecification,FITTING_EQUATIONS_SET_FIELD_USER_NUMBER,fittingEquationsSetField)
fittingEquationsSet.CreateFinish()

#-----------------------------------------------------------------------------------------------------------
# FITTING DEPENDENT FIELD
#-----------------------------------------------------------------------------------------------------------

# Create the fitting dependent field
fittingDependentField = oc.Field()
fittingEquationsSet.DependentCreateStart(FITTING_DEPENDENT_FIELD_USER_NUMBER,fittingDependentField)
fittingDependentField.VariableLabelSet(oc.FieldVariableTypes.U,"FittingU")
# Set the number of components to be such to fit the stress and lambda fields
fittingDependentField.NumberOfComponentsSet(oc.FieldVariableTypes.U,2*NUMBER_OF_VOIGT_COMPONENTS)
for componentIdx in range(1,2*NUMBER_OF_VOIGT_COMPONENTS+1):
    # Set the field variables to be trilinear Lagrange
    fittingDependentField.ComponentMeshComponentSet(oc.FieldVariableTypes.U,componentIdx,TRILINEAR_LAGRANGE_MESH_COMPONENT)
# Finish creating the fitting dependent field
fittingEquationsSet.DependentCreateFinish()

#-----------------------------------------------------------------------------------------------------------
# FITTING INDEPENDENT FIELD
#-----------------------------------------------------------------------------------------------------------
# Create the fitting independent field
fittingIndependentField = oc.Field()
fittingEquationsSet.IndependentCreateStart(FITTING_INDEPENDENT_FIELD_USER_NUMBER,fittingIndependentField)
fittingIndependentField.VariableLabelSet(oc.FieldVariableTypes.U,"GaussLambda")
fittingIndependentField.VariableLabelSet(oc.FieldVariableTypes.V,"LambdaWeight")
# Set the number of components to fit the stress and lambda fields
fittingIndependentField.NumberOfComponentsSet(oc.FieldVariableTypes.U,2*NUMBER_OF_VOIGT_COMPONENTS)
fittingIndependentField.NumberOfComponentsSet(oc.FieldVariableTypes.V,2*NUMBER_OF_VOIGT_COMPONENTS)
# Finish creating the fitting independent field
fittingEquationsSet.IndependentCreateFinish()

# Initialise data point weight field to 1.0
for componentIdx in range(1,2*NUMBER_OF_VOIGT_COMPONENTS+1):
    fittingIndependentField.ComponentValuesInitialiseDP(oc.FieldVariableTypes.V,
                                                        oc.FieldParameterSetTypes.VALUES,componentIdx,1.0)
    
#-----------------------------------------------------------------------------------------------------------
# FITTING MATERIAL FIELD
#-----------------------------------------------------------------------------------------------------------

# Create material field (Sobolev parameters)
fittingMaterialField = oc.Field()
fittingEquationsSet.MaterialsCreateStart(FITTING_MATERIALS_FIELD_USER_NUMBER,fittingMaterialField)
fittingMaterialField.VariableLabelSet(oc.FieldVariableTypes.U,"SmoothingParameters")
fittingEquationsSet.MaterialsCreateFinish()

# Set kappa and tau - Sobolev smoothing parameters
fittingMaterialField.ComponentValuesInitialiseDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,1,TAU)
fittingMaterialField.ComponentValuesInitialiseDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,2,KAPPA)
fittingMaterialField.ComponentValuesInitialiseDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,3,TAU)
fittingMaterialField.ComponentValuesInitialiseDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,4,KAPPA)
fittingMaterialField.ComponentValuesInitialiseDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,5,KAPPA)
fittingMaterialField.ComponentValuesInitialiseDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,6,TAU)
fittingMaterialField.ComponentValuesInitialiseDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,7,KAPPA)
fittingMaterialField.ComponentValuesInitialiseDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,8,KAPPA)
fittingMaterialField.ComponentValuesInitialiseDP(oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,9,KAPPA)

#-----------------------------------------------------------------------------------------------------------
# FITTING EQUATIONS
#-----------------------------------------------------------------------------------------------------------

# Create the fitting equations
fittingEquations = oc.Equations()
fittingEquationsSet.EquationsCreateStart(fittingEquations)
# Set the fitting equations sparsity type
fittingEquations.SparsityTypeSet(oc.EquationsSparsityTypes.SPARSE)
# Set the fitting equations output type to none
fittingEquations.OutputTypeSet(oc.EquationsOutputTypes.NONE)
# Finish creating the fitting equations
fittingEquationsSet.EquationsCreateFinish()

#-----------------------------------------------------------------------------------------------------------
# FITTING PROBLEM
#-----------------------------------------------------------------------------------------------------------

# Create fitting problem
fittingProblemSpecification = [oc.ProblemClasses.FITTING,
                        oc.ProblemTypes.FITTING,
                        oc.ProblemSubtypes.STATIC_LINEAR_FITTING]
fittingProblem = oc.Problem()
fittingProblem.CreateStart(FITTING_PROBLEM_USER_NUMBER,context,fittingProblemSpecification)
fittingProblem.CreateFinish()

#-----------------------------------------------------------------------------------------------------------
# FITTING CONTROL LOOPS
#-----------------------------------------------------------------------------------------------------------

# Create control loops
fittingProblem.ControlLoopCreateStart()
fittingProblem.ControlLoopCreateFinish()

#-----------------------------------------------------------------------------------------------------------
# FITTING SOLVERS
#-----------------------------------------------------------------------------------------------------------

# Create problem solver
fittingSolver = oc.Solver()
fittingProblem.SolversCreateStart()
fittingProblem.SolverGet([oc.ControlLoopIdentifiers.NODE],1,fittingSolver)
fittingSolver.OutputTypeSet(oc.SolverOutputTypes.PROGRESS)
fittingSolver.LinearIterativeMaximumIterationsSet(1000000)
fittingProblem.SolversCreateFinish()

#-----------------------------------------------------------------------------------------------------------
# FITTING SOLVER EQUATIONS
#-----------------------------------------------------------------------------------------------------------

# Create fitting solver equations and add fitting equations set to solver equations
fittingSolverEquations = oc.SolverEquations()
fittingProblem.SolverEquationsCreateStart()
# Get the solver equations
fittingSolver.SolverEquationsGet(fittingSolverEquations)
fittingSolverEquations.SparsityTypeSet(oc.SolverEquationsSparsityTypes.SPARSE)
fittingEquationsSetIndex = fittingSolverEquations.EquationsSetAdd(fittingEquationsSet)
fittingProblem.SolverEquationsCreateFinish()

#-----------------------------------------------------------------------------------------------------------
# FITTING BOUNDARY CONDITIONS
#-----------------------------------------------------------------------------------------------------------

# Prescribe boundary conditions for the fitting problem
fittingBoundaryConditions = oc.BoundaryConditions()
fittingSolverEquations.BoundaryConditionsCreateStart(fittingBoundaryConditions)

# For the stress field fitting components set the last node (which is free) to have zero stress
nodeNumber = NUMBER_OF_NODES
nodeDomain = decomposition.NodeDomainGet(TRILINEAR_LAGRANGE_MESH_COMPONENT,nodeNumber)
if nodeDomain == computationalNodeNumber:
    for componentIdx in range(1,NUMBER_OF_VOIGT_COMPONENTS+1):
        fittingBoundaryConditions.SetNode(fittingDependentField,oc.FieldVariableTypes.U,
	                                  1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,componentIdx,
                                          oc.BoundaryConditionsTypes.FIXED,0.0)

# For the lambda field fitting components set the first node (which is fixed) to have a lambda of 1/0
nodeNumber = 1
nodeDomain = decomposition.NodeDomainGet(TRILINEAR_LAGRANGE_MESH_COMPONENT,nodeNumber)
if nodeDomain == computationalNodeNumber:
    for componentIdx in range(NUMBER_OF_VOIGT_COMPONENTS+1,NUMBER_OF_VOIGT_COMPONENTS+NUMBER_OF_DIMENSIONS+1):
        fittingBoundaryConditions.SetNode(fittingDependentField,oc.FieldVariableTypes.U,
	                                  1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,componentIdx,
                                          oc.BoundaryConditionsTypes.FIXED,1.0)
    for componentIdx in range(NUMBER_OF_VOIGT_COMPONENTS+NUMBER_OF_DIMENSIONS+1,2*NUMBER_OF_VOIGT_COMPONENTS+1):
        fittingBoundaryConditions.SetNode(fittingDependentField,oc.FieldVariableTypes.U,
	                                  1,oc.GlobalDerivativeConstants.NO_GLOBAL_DERIV,nodeNumber,componentIdx,
                                          oc.BoundaryConditionsTypes.FIXED,0.0)
        
fittingSolverEquations.BoundaryConditionsCreateFinish()

#-----------------------------------------------------------------------------------------------------------
# ELASTICITY GROWTH AND FITTING MAIN WORKFLOW
#-----------------------------------------------------------------------------------------------------------

# Loop over the time steps
time = START_TIME

timeString = format(time)
filename = "HeartTubeGrowth_"+timeString
fields = oc.Fields()
fields.CreateRegion(region)
fields.NodesExport(filename,"FORTRAN")
fields.ElementsExport(filename,"FORTRAN")

currentLength=0.0
while time <= STOP_TIME:

    # Set the times
    elasticityProblemLoop.TimesSet(time,time+DELTA_TIME,DELTA_TIME)

    # Solve the elastic growth problem
    elasticityProblem.Solve()

    # Evaluate the derived fields
    elasticityEquationsSet.DerivedVariableCalculate(oc.EquationsSetDerivedTensorTypes.DEFORMATION_GRADIENT)
    elasticityEquationsSet.DerivedVariableCalculate(oc.EquationsSetDerivedTensorTypes.DEFORMATION_GRADIENT_FIBRE)
    elasticityEquationsSet.DerivedVariableCalculate(oc.EquationsSetDerivedTensorTypes.R_CAUCHY_GREEN_DEFORMATION)
    elasticityEquationsSet.DerivedVariableCalculate(oc.EquationsSetDerivedTensorTypes.CAUCHY_STRESS)
    elasticityEquationsSet.DerivedVariableCalculate(oc.EquationsSetDerivedTensorTypes.CAUCHY_STRESS_FIBRE)

    # Compute the principal stretches to fit (and to constrain the growth)
    for wallElementIdx in range(1,NUMBER_OF_WALL_ELEMENTS+1):
        for lengthElementIdx in range(1,NUMBER_OF_LENGTH_ELEMENTS+1):
            for circumfrentialElementIdx in range(1,NUMBER_OF_CIRCUMFRENTIAL_ELEMENTS+1):
                elementNumber = circumfrentialElementIdx + (lengthElementIdx-1)*NUMBER_OF_CIRCUMFRENTIAL_ELEMENTS + \
                    (wallElementIdx-1)*NUMBER_OF_CIRCUMFRENTIAL_ELEMENTS*NUMBER_OF_LENGTH_ELEMENTS
                elementDomain = decomposition.ElementDomainGet(elementNumber)
                if elementDomain == computationalNodeNumber:
                    for xiIdx3 in range(1,NUMBER_OF_GAUSS_XI+1):
                        for xiIdx2 in range(1,NUMBER_OF_GAUSS_XI+1):
                            for xiIdx1 in range(1,NUMBER_OF_GAUSS_XI+1):
                                gaussPointNumber = xiIdx1 + (xiIdx2-1)*NUMBER_OF_GAUSS_XI + \
                                    (xiIdx3-1)*NUMBER_OF_GAUSS_XI*NUMBER_OF_GAUSS_XI
                                
                                print("")
                                print("Element number = ",elementNumber,", Gauss point number = ",gaussPointNumber)
                                
                                # Get the components of the fibre deformation gradient tensor at the Gauss point
                                FNu11 = elasticityDerivedField.ParameterSetGetGaussPointDP(oc.FieldVariableTypes.U2,
                                                                                           oc.FieldParameterSetTypes.VALUES,
                                                                                           gaussPointNumber,elementNumber,
                                                                                           TENSOR11_COMPONENT)
                                FNu21 = elasticityDerivedField.ParameterSetGetGaussPointDP(oc.FieldVariableTypes.U2,
                                                                                           oc.FieldParameterSetTypes.VALUES,
                                                                                           gaussPointNumber,elementNumber,
                                                                                           TENSOR21_COMPONENT)
                                FNu31 = elasticityDerivedField.ParameterSetGetGaussPointDP(oc.FieldVariableTypes.U2,
                                                                                           oc.FieldParameterSetTypes.VALUES,
                                                                                           gaussPointNumber,elementNumber,
                                                                                           TENSOR31_COMPONENT)
                                FNu12 = elasticityDerivedField.ParameterSetGetGaussPointDP(oc.FieldVariableTypes.U2,
                                                                                           oc.FieldParameterSetTypes.VALUES,
                                                                                           gaussPointNumber,elementNumber,
                                                                                           TENSOR12_COMPONENT)
                                FNu22 = elasticityDerivedField.ParameterSetGetGaussPointDP(oc.FieldVariableTypes.U2,
                                                                                           oc.FieldParameterSetTypes.VALUES,
                                                                                           gaussPointNumber,elementNumber,
                                                                                           TENSOR22_COMPONENT)
                                FNu32 = elasticityDerivedField.ParameterSetGetGaussPointDP(oc.FieldVariableTypes.U2,
                                                                                           oc.FieldParameterSetTypes.VALUES,
                                                                                           gaussPointNumber,elementNumber,
                                                                                           TENSOR32_COMPONENT)
                                FNu13 = elasticityDerivedField.ParameterSetGetGaussPointDP(oc.FieldVariableTypes.U2,
                                                                                           oc.FieldParameterSetTypes.VALUES,
                                                                                           gaussPointNumber,elementNumber,
                                                                                           TENSOR13_COMPONENT)
                                FNu23 = elasticityDerivedField.ParameterSetGetGaussPointDP(oc.FieldVariableTypes.U2,
                                                                                           oc.FieldParameterSetTypes.VALUES,
                                                                                           gaussPointNumber,elementNumber,
                                                                                           TENSOR23_COMPONENT)
                                FNu33 = elasticityDerivedField.ParameterSetGetGaussPointDP(oc.FieldVariableTypes.U2,
                                                                                           oc.FieldParameterSetTypes.VALUES,
                                                                                           gaussPointNumber,elementNumber,
                                                                                           TENSOR33_COMPONENT)
                                FNu = np.array([[ FNu11, FNu12, FNu13 ],
                                                [ FNu21, FNu22, FNu23 ],
                                                [ FNu31, FNu32, FNu33 ]])
                                # Compute the polar decomposition to find the stretches. Use spatial coordinates
                                RNu, VNu = linalg.polar(FNu,side='right')
                                stretchf=VNu[0,0]
                                stretchs=VNu[1,1]
                                stretchn=VNu[2,2]
                                stretchfs=VNu[0,1]
                                stretchfn=VNu[0,2]
                                stretchsn=VNu[1,2]
                                print("Stretch F  = ",stretchf)
                                print("Stretch S  = ",stretchs)
                                print("Stretch N  = ",stretchn)
                                print("Stretch FS = ",stretchfs)
                                print("Stretch FN = ",stretchfn)
                                print("Stretch SN = ",stretchsn)
                                # Update the current stretches in the fitted independent field
                                fittingIndependentField.ParameterSetAddGaussPointDP(oc.FieldVariableTypes.U,
                                                                                    oc.FieldParameterSetTypes.VALUES,
                                                                                    gaussPointNumber,elementNumber,
                                                                                    NUMBER_OF_VOIGT_COMPONENTS+VOIGT11_COMPONENT,
                                                                                    stretchf-1.0)
                                fittingIndependentField.ParameterSetAddGaussPointDP(oc.FieldVariableTypes.U,
                                                                                    oc.FieldParameterSetTypes.VALUES,
                                                                                    gaussPointNumber,elementNumber,
                                                                                    NUMBER_OF_VOIGT_COMPONENTS+VOIGT22_COMPONENT,
                                                                                    stretchs-1.0)
                                fittingIndependentField.ParameterSetAddGaussPointDP(oc.FieldVariableTypes.U,
                                                                                    oc.FieldParameterSetTypes.VALUES,
                                                                                    gaussPointNumber,elementNumber,
                                                                                    NUMBER_OF_VOIGT_COMPONENTS+VOIGT33_COMPONENT,
                                                                                    stretchn-1.0)
                                fittingIndependentField.ParameterSetAddGaussPointDP(oc.FieldVariableTypes.U,
                                                                                    oc.FieldParameterSetTypes.VALUES,
                                                                                    gaussPointNumber,elementNumber,
                                                                                    NUMBER_OF_VOIGT_COMPONENTS+VOIGT12_COMPONENT,
                                                                                    stretchfs)
                                fittingIndependentField.ParameterSetAddGaussPointDP(oc.FieldVariableTypes.U,
                                                                                    oc.FieldParameterSetTypes.VALUES,
                                                                                    gaussPointNumber,elementNumber,
                                                                                    NUMBER_OF_VOIGT_COMPONENTS+VOIGT13_COMPONENT,
                                                                                    stretchfn)
                                fittingIndependentField.ParameterSetAddGaussPointDP(oc.FieldVariableTypes.U,
                                                                                    oc.FieldParameterSetTypes.VALUES,
                                                                                    gaussPointNumber,elementNumber,
                                                                                    NUMBER_OF_VOIGT_COMPONENTS+VOIGT23_COMPONENT,
                                                                                    stretchsn)
                                # Update the beta parameters if required
                                if GROWTH_MODEL == LIMITED_GROWTH_MODEL:
                                    growthCellMLParametersField.ParameterSetAddGaussPointDP(oc.FieldVariableTypes.U,
                                                                                               oc.FieldParameterSetTypes.VALUES,
                                                                                               gaussPointNumber,elementNumber,
                                                                                               fibreBetaComponent,stretchf-1.0)
                                    growthCellMLParametersField.ParameterSetAddGaussPointDP(oc.FieldVariableTypes.U,
                                                                                               oc.FieldParameterSetTypes.VALUES,
                                                                                               gaussPointNumber,elementNumber,
                                                                                               sheetBetaComponent,stretchs-1.0)
                                    growthCellMLParametersField.ParameterSetAddGaussPointDP(oc.FieldVariableTypes.U,
                                                                                               oc.FieldParameterSetTypes.VALUES,
                                                                                               gaussPointNumber,elementNumber,
                                                                                               normalBetaComponent,stretchn-1.0)
                                    growthCellMLParametersField.ParameterSetAddGaussPointDP(oc.FieldVariableTypes.U,
                                                                                               oc.FieldParameterSetTypes.VALUES,
                                                                                               gaussPointNumber,elementNumber,
                                                                                               fibreSheetBetaComponent,stretchfs)
                                    growthCellMLParametersField.ParameterSetAddGaussPointDP(oc.FieldVariableTypes.U,
                                                                                               oc.FieldParameterSetTypes.VALUES,
                                                                                               gaussPointNumber,elementNumber,
                                                                                               fibreNormalBetaComponent,stretchfn)
                                    growthCellMLParametersField.ParameterSetAddGaussPointDP(oc.FieldVariableTypes.U,
                                                                                               oc.FieldParameterSetTypes.VALUES,
                                                                                               gaussPointNumber,elementNumber,
                                                                                               sheetNormalBetaComponent,stretchsn)
                                    

    # Update the Cauchy fibre stress in the fitting independent field for the fitting problem
    oc.Field.ParametersToFieldParametersComponentCopy(
        elasticityDerivedField,oc.FieldVariableTypes.U4,oc.FieldParameterSetTypes.VALUES,VOIGT11_COMPONENT,
        fittingIndependentField,oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,VOIGT11_COMPONENT)
    oc.Field.ParametersToFieldParametersComponentCopy(
        elasticityDerivedField,oc.FieldVariableTypes.U4,oc.FieldParameterSetTypes.VALUES,VOIGT22_COMPONENT,
        fittingIndependentField,oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,VOIGT22_COMPONENT)
    oc.Field.ParametersToFieldParametersComponentCopy(
        elasticityDerivedField,oc.FieldVariableTypes.U4,oc.FieldParameterSetTypes.VALUES,VOIGT33_COMPONENT,
        fittingIndependentField,oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,VOIGT33_COMPONENT)
    oc.Field.ParametersToFieldParametersComponentCopy(
        elasticityDerivedField,oc.FieldVariableTypes.U4,oc.FieldParameterSetTypes.VALUES,VOIGT12_COMPONENT,
        fittingIndependentField,oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,VOIGT12_COMPONENT)
    oc.Field.ParametersToFieldParametersComponentCopy(
        elasticityDerivedField,oc.FieldVariableTypes.U4,oc.FieldParameterSetTypes.VALUES,VOIGT13_COMPONENT,
        fittingIndependentField,oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,VOIGT13_COMPONENT)
    oc.Field.ParametersToFieldParametersComponentCopy(
        elasticityDerivedField,oc.FieldVariableTypes.U4,oc.FieldParameterSetTypes.VALUES,VOIGT23_COMPONENT,
        fittingIndependentField,oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,VOIGT23_COMPONENT)

    # Solve the fitting problem
    fittingProblem.Solve()

    # Copy the fitting dependent field to the stress field
    oc.Field.ParametersToFieldParametersComponentCopy(
        fittingDependentField,oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,VOIGT11_COMPONENT,
        stressField,oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,VOIGT11_COMPONENT)
    oc.Field.ParametersToFieldParametersComponentCopy(
        fittingDependentField,oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,VOIGT22_COMPONENT,
        stressField,oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,VOIGT22_COMPONENT)
    oc.Field.ParametersToFieldParametersComponentCopy(
        fittingDependentField,oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,VOIGT33_COMPONENT,
        stressField,oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,VOIGT33_COMPONENT)
    oc.Field.ParametersToFieldParametersComponentCopy(
        fittingDependentField,oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,VOIGT12_COMPONENT,
        stressField,oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,VOIGT12_COMPONENT)
    oc.Field.ParametersToFieldParametersComponentCopy(
        fittingDependentField,oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,VOIGT13_COMPONENT,
        stressField,oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,VOIGT13_COMPONENT)
    oc.Field.ParametersToFieldParametersComponentCopy(
        fittingDependentField,oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,VOIGT23_COMPONENT,
        stressField,oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,VOIGT23_COMPONENT)
    
    # Copy the fitting dependent field to the lambda field
    for componentIdx in range(1,NUMBER_OF_VOIGT_COMPONENTS+1):
        oc.Field.ParametersToFieldParametersComponentCopy(
            fittingDependentField,oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,NUMBER_OF_VOIGT_COMPONENTS+componentIdx,
            lambdaField,oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,componentIdx)

    # Export results
    timeString = format(time)
    filename = "HeartTubeGrowth_"+timeString
    fields.NodesExport(filename,"FORTRAN")
    fields.ElementsExport(filename,"FORTRAN")

    # Set the geometric field to the current deformed geometry
    for componentIdx in range(1,NUMBER_OF_DIMENSIONS+1):
        oc.Field.ParametersToFieldParametersComponentCopy(
            elasticityDependentField,oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,componentIdx,
            geometricField,oc.FieldVariableTypes.U,oc.FieldParameterSetTypes.VALUES,componentIdx)

    # Reset growth state field to 1.0/0.0        
    oc.Field.ComponentValuesInitialiseDP(growthCellMLStateField,oc.FieldVariableTypes.U,
                                         oc.FieldParameterSetTypes.VALUES,LAMBDA_F_COMPONENT,1.0)
    oc.Field.ComponentValuesInitialiseDP(growthCellMLStateField,oc.FieldVariableTypes.U,
                                         oc.FieldParameterSetTypes.VALUES,LAMBDA_S_COMPONENT,1.0)
    oc.Field.ComponentValuesInitialiseDP(growthCellMLStateField,oc.FieldVariableTypes.U,
                                         oc.FieldParameterSetTypes.VALUES,LAMBDA_N_COMPONENT,1.0)
    oc.Field.ComponentValuesInitialiseDP(growthCellMLStateField,oc.FieldVariableTypes.U,
                                         oc.FieldParameterSetTypes.VALUES,LAMBDA_FS_COMPONENT,0.0)
    oc.Field.ComponentValuesInitialiseDP(growthCellMLStateField,oc.FieldVariableTypes.U,
                                         oc.FieldParameterSetTypes.VALUES,LAMBDA_FN_COMPONENT,0.0)
    oc.Field.ComponentValuesInitialiseDP(growthCellMLStateField,oc.FieldVariableTypes.U,
                                         oc.FieldParameterSetTypes.VALUES,LAMBDA_SN_COMPONENT,0.0)

    # Increment time
    time = time + DELTA_TIME
    
#-----------------------------------------------------------------------------------------------------------
# FINALISE AND CLEANUP
#-----------------------------------------------------------------------------------------------------------
   
fields.Finalise()
context.Destroy()

