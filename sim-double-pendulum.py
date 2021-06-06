import pinocchio as pin
import hppfcl as fcl
import numpy as np
import matplotlib.pyplot as plt
from math import *
import time
import sys


def deg(arg):
    return degrees(arg)


DoF = 2  # number of pendulums
model = pin.Model()
geom_model = pin.GeometryModel()

parent_id = 0
joint_placement = pin.SE3.Identity()
body_mass = 3.50
body_radius = 0.1

shape0 = fcl.Sphere(body_radius)
geom0_obj = pin.GeometryObject("base", 0, shape0, pin.SE3.Identity())
geom0_obj.meshColor = np.array([1., 0.2, 0.8, 1.0])
geom_model.addGeometryObject(geom0_obj)

# Model geometry construction:
for k in range(DoF):
    joint_name = "joint_" + str(k + 1)
    joint_id = model.addJoint(parent_id, pin.JointModelRY(), joint_placement, joint_name)

    den = k + 1.0
    body_inertia = pin.Inertia.FromSphere(body_mass / den, body_radius)  # second link with less inertia
    body_placement = joint_placement.copy()
    body_placement.translation[2] = 1.
    model.appendBodyToJoint(joint_id, body_inertia, body_placement)

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

from pinocchio.visualize import GepettoVisualizer

visual_model = geom_model
viz = GepettoVisualizer(model, geom_model, visual_model)

# Initialize the viewer.
try:
    viz.initViewer()
except ImportError as err:
    print("Error while initializing the viewer. It seems you should install gepetto-viewer")
    print(err)
    sys.exit(0)

try:
    viz.loadViewerModel("pinocchio")
except AttributeError as err:
    print("Error while loading the viewer model. It seems you should start gepetto-viewer")
    print(err)
    sys.exit(0)

viz.sceneName = "Double Pendulum Leg"
# Display a robot configuration.
q0 = pin.neutral(model)
viz.display(q0)

# Simulation Config:
dt = 0.001  # running simulation at 1000 Hz
dt = 0.008
sim_duration = 7  # simulation time period in sec
sim_steps = floor(sim_duration / dt)

# Model properties
model.lowerPositionLimit.fill(-pi)
model.upperPositionLimit.fill(+pi)
# model.damping = np.array([0.0, 30.0])
model.friction = np.array([0.0, 30.0])
print("model damping: " + str(model.friction))

# Controller (PD):
ctrl_type = 'pdg'
Kp = np.eye(model.nv) * 125
Kd = np.eye(model.nv) * 18
if ctrl_type == 'id':
    Kp = 0.18 * Kp
    Kd = 0.44 * Kd
# print(Kp)
# Input
input_type = 'sin'
freqs = np.array([.0, 0.8])
amps  = np.array([0.2, 0.1])

# Desired states variables
q_des = np.array([pi * (178 / 180), pi * (90 / 180)])
dq_des = np.zeros(model.nv)
ddq_des = np.ones(model.nv) * 0.1

# q = pin.randomConfiguration(model)
# Initial states, q0, dq0
q = np.array([pi * (135 / 180), pi * (15 / 180)])
dq = np.zeros(model.nv)
q0 = q
dq0 = dq
# Auxiliar state variables for integration
dq_last = np.zeros(model.nv)
ddq_last = dq_last.copy()

# Logging variables
downsmpl_log = 0
q_log = np.empty([1, 1 + model.nq]) * nan
dq_log = q_log.copy()
ddq_log = q_log.copy()

data_sim = model.createData()
t = 0.00
# SIMULATION:
for k in range(sim_steps):

    tau_control = np.zeros((model.nv))
    pin.computeAllTerms(model, data_sim, q, dq)
    Mq = data_sim.M
    hq = data_sim.C
    grav = data_sim.g

    # Step reference at t=9s
    # if t > 3.0:
    #    q_des = np.array([math.pi*(80/180), math.pi*(65/180)])

    # Oscillatory reference
    q_des = q0 + np.array([0.02 * sin(2 * 3.1415 * 0.0 * t), .2 * sin(2 * 3.1415 * 0.8 * t)])

    # PD Control
    if ctrl_type == 'pd':
        tau_control = Kp.dot(q_des - q) + Kd.dot(dq_des - dq)
    # PD+Feedforward Control
    if ctrl_type == 'pdff':
        tau_control = Kp.dot(q_des - q) + Kd.dot(dq_des - dq) + np.multiply(np.diag(Mq), ddq_des)
    # PD Control + Grav Compensation
    if ctrl_type == 'pdg':
        tau_control = Kp.dot(q_des - q) + Kd.dot(dq_des - dq) + grav
    # Inverse Dynamics Control, data_sim.nle contains C + g
    if ctrl_type == 'id':
        tau_control = Mq.dot(ddq_des + Kp.dot(q_des - q) + Kd.dot(dq_des - dq)) + data_sim.nle

    ddq = pin.aba(model, data_sim, q, dq, tau_control)  # Forward dynamics

    # Forward Euler Integration with Trapeziodal Rule
    dq += (ddq_last + ddq) * dt * 0.5
    ddq_last = ddq
    q += (dq_last + dq) * dt * 0.5
    dq_last = dq
    # q = pin.integrate(model,q,dq*dt)

    # Log variables
    downsmpl_log += 1
    if downsmpl_log > 1:
        # q_log = np.vstack
        q_log = np.vstack((q_log, [t, deg(q[0]), deg(q[1])]))
        dq_log = np.vstack((dq_log, [t, deg(dq[0]), deg(dq[1])]))
        ddq_log = np.vstack((ddq_log, [t, deg(ddq[0]), deg(ddq[1])]))
        downsmpl_log = 0

    viz.display(q)
    time.sleep(dt)
    t += dt

# positions = data_sim.p
# velocities = data_sim.v # works, why?

print("Simulation ended, here comes the plots...")
plt.plot(q_log[:, 0], q_log[:, 1])
plt.plot(q_log[:, 0], q_log[:, 2])
plt.show()

# plt.plot(dq_log[:,0], dq_log[:,1])
# plt.plot(dq_log[:,0], dq_log[:,2])
# plt.show()
# FINALLY!!!
