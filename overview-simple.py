# My first Pinocchio tutorial
# github/bitbucket @qleonardolp - 2021

from __future__ import print_function
import pinocchio as pino

# Code:
model = pino.buildSampleModelManipulator()
data = model.createData()
q = pino.randomConfiguration(model)
print('q0: ', q.T)
v = pino.utils.zero(model.nv)
a = pino.utils.zero(model.nv)
tau = pino.rnea(model, data, q, v, a)
print('torques = ', tau.T)
