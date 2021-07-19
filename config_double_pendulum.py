import pinocchio as pin
import hppfcl as fcl
import numpy as np
from math import *
import time
import sys

from pinocchio.visualize import GepettoVisualizer

# From "Adjustments to McConville et al. and Young et al. body segment inertial parameters"
# Using 70 Kg Female adult, Thigh and Leg:
hum_body_mass = np.array([0.146*70, 0.048*70])

DoF = 2  # number of pendulums
Model = pin.Model()
geom_model = pin.GeometryModel()

parent_id = 0
joint_placement = pin.SE3.Identity()
#print(joint_placement.translation)
exo_body_mass = 2*hum_body_mass
body_radius = 0.1

shape0 = fcl.Sphere(body_radius)
geom0_obj = pin.GeometryObject("base", 0, shape0, pin.SE3.Identity())
geom0_obj.meshColor = np.array([1., 0.2, 0.8, 1.0])
geom_model.addGeometryObject(geom0_obj)

# Model geometry construction:
for k in range(DoF):
    joint_name = "joint_" + str(k + 1)
    joint_id = Model.addJoint(parent_id, pin.JointModelRY(), joint_placement, joint_name)
    #Model.addJointFrame(joint_id)

    den = k + 1.0
    body_inertia = pin.Inertia.FromSphere(exo_body_mass[k], body_radius)  # second link with less inertia
    body_placement = joint_placement.copy()
    body_placement.translation[2] = 1.
    Model.appendBodyToJoint(joint_id, body_inertia, body_placement)

    geom1_name = "ball_" + str(k + 1)
    shape1 = fcl.Sphere(body_radius)
    geom1_obj = pin.GeometryObject(geom1_name, joint_id, shape1, body_placement)
    geom1_obj.meshColor = np.ones((4))
    geom_model.addGeometryObject(geom1_obj)

    geom2_name = "bar_" + str(k + 1)
    shape2 = fcl.Cylinder(body_radius / 4., body_placement.translation[2])
    shape2_placement = body_placement.copy()
    shape2_placement.translation[2] /= 2.

    geom2_obj = pin.GeometryObject(geom2_name, joint_id, shape2, shape2_placement)
    geom2_obj.meshColor = np.array([0., 0., 0., 1.])
    geom_model.addGeometryObject(geom2_obj)

    parent_id = joint_id
    joint_placement = body_placement.copy()
# Add Gepetto handler
visual_model = geom_model
viz = GepettoVisualizer(Model, geom_model, visual_model)


# Human Leg model:
humModel = pin.Model()
humGeom = pin.GeometryModel()

parent_id = 0
#base_placement = pin.XYZQUATToSE3([.0, .23, .0, 1.0, .0, .0, .0])
base_placement = pin.SE3.Identity()
joint_placement = pin.SE3.Identity()
body_radius = 0.07

shape0 = fcl.Sphere(body_radius)
geom0_obj = pin.GeometryObject("base", 0, shape0, base_placement)
geom0_obj.meshColor = np.array([1., 0.2, 0.8, .23])
#geom0_obj.placement.translation[1] = .23
humGeom.addGeometryObject(geom0_obj)

# Model geometry construction:
for k in range(DoF):
    joint_name = "joint_" + str(k + 1)
    joint_id = humModel.addJoint(parent_id, pin.JointModelRY(), joint_placement, joint_name)
    #Model.addJointFrame(joint_id)

    body_inertia = pin.Inertia.FromSphere(hum_body_mass[k], body_radius)  # second link with less inertia
    body_placement = joint_placement.copy()
    body_placement.translation[2] = 1.
    humModel.appendBodyToJoint(joint_id, body_inertia, body_placement)

    geom1_name = "ball_" + str(k + 1)
    shape1 = fcl.Sphere(body_radius)
    geom1_obj = pin.GeometryObject(geom1_name, joint_id, shape1, body_placement)
    geom1_obj.meshColor = np.ones((4))
    humGeom.addGeometryObject(geom1_obj)

    geom2_name = "bar_" + str(k + 1)
    shape2 = fcl.Cylinder(body_radius / 4., body_placement.translation[2])
    shape2_placement = body_placement.copy()
    shape2_placement.translation[2] /= 2.

    geom2_obj = pin.GeometryObject(geom2_name, joint_id, shape2, shape2_placement)
    geom2_obj.meshColor = np.array([0., 0., 0., .23])
    humGeom.addGeometryObject(geom2_obj)

    parent_id = joint_id
    joint_placement = body_placement.copy()
# Gepetto handler
humVisual = humGeom
viz_hum = GepettoVisualizer(humModel, humGeom, humVisual)

# Initialize the viewer.
try:
    viz.initViewer()
    viz_hum.initViewer()
except ImportError as err:
    print("Error while initializing the viewer. It seems you should install gepetto-viewer")
    print(err)
    sys.exit(0)

try:
    viz.loadViewerModel("robot")
    viz_hum.loadViewerModel("human")
except AttributeError as err:
    print("Error while loading the viewer model. It seems you should start gepetto-viewer")
    print(err)
    sys.exit(0)

viz.sceneName = "Double Pendulum Leg"
# Display a robot configuration.
#q0 = pin.neutral(Model)
#viz.display(q0)

# Simulation Config:
dt = 0.001  # running simulation at 1000 Hz
dt = 0.008
sim_duration = 20  # simulation time period in sec
step_input_time = sim_duration/2
sim_steps = floor(sim_duration / dt)

humStiffness = np.eye(humModel.nv) * 850.0
humDamping = np.eye(humModel.nv) * 12.0 # cuidado, nao pode ser muito alto.

# Model properties (should be set before Gepetto)
#Model.lowerPositionLimit.fill(-pi)
#Model.upperPositionLimit.fill(+pi)
#Model.damping = np.array([6.0, 30.0, 5.0, 5.]) # ???
#Model.friction = np.array([0.0, 30.0])