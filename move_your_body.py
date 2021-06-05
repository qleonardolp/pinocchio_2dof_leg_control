# First tutorial from practical exercises in Pinocchio Documentation
from pinocchio.utils import *
import pinocchio as pino
import numpy as np

A = np.matrix([[1, 2, 3, 4], [2, 4, 6, 8]])
b = np.zeros([4, 1])
c = A * b

C = eye(len(c)) * c
# print(C)  # show C matrix

# Import Robot
# URDF_PATH = '/opt/openrobots/share/ur5_description/urdf/ur5_gripper.urdf'
URDF_PATH = '/opt/openrobots/share/example-robot-data/robots/double_pendulum_description/urdf/double_pendulum.urdf'
robot_model = pino.buildModelFromUrdf(URDF_PATH)
# robot_model = RobotWrapper.BuildFromURDF(UR5_URDF) # arrombado que nao funciona

print(robot_model.name + " URDF model loaded.")

# exploring the model
# Create data required by the algorithms
data = robot_model.createData()
# Sample a random configuration
q = pino.randomConfiguration(robot_model)
print('q: %s' % q.T)
# Perform the forward kinematics over the kinematic tree
pino.forwardKinematics(robot_model,data,q)
# Print out the placement of each joint of the kinematic tree
for name, oMi in zip(robot_model.names, data.oMi):
    print(("{:<24} : {: .2f} {: .2f} {: .2f}" .format( name, *oMi.translation.T.flat )))

