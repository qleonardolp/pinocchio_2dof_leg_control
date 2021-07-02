import pinocchio as pin
import numpy as np
from math import *
import config_double_pendulum as conf

Romega = np.array([[1., .0], [.0, 1.]])
Rpeaks = np.array([[.7, .0], [.0, .5]])
Rdc    = np.array([[1.6, .0], [.0, 1.4]])

# See toDynamicParameters() definition at
# https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/classpinocchio_1_1InertiaTpl.html
humThighJyy = conf.humModel.inertias[1].toDynamicParameters()[6]
humShankJyy = conf.humModel.inertias[2].toDynamicParameters()[6]

hum_inertia = np.array([[humThighJyy, .0], [.0, humShankJyy]])
hum_stiff   = conf.humStiffness
hum_damp    = conf.humDamping

hum_omega_n = np.sqrt(hum_stiff * np.linalg.inv(hum_inertia))
sqrtJK = np.sqrt(hum_inertia*hum_stiff)
hum_zetas = hum_damp * (np.linalg.inv(2*sqrtJK))
rho = Rdc * np.linalg.inv(Rpeaks) * hum_zetas * np.sqrt(np.eye(2) - hum_zetas*hum_zetas)

I_des = hum_inertia * np.linalg.inv(Rdc * Romega*Romega)
omg_des = np.diag(Romega * hum_omega_n)
zeta_des = np.sqrt( 0.5*np.eye(2) - 0.5*np.sqrt(np.eye(2) - 4*rho*rho) )

# From the Integral Admittance definition, the desired Impedance Parameters are:
imp_kp = I_des * np.square(omg_des)
imp_kd = I_des * (2 * zeta_des * omg_des)

print(imp_kp)
print(imp_kd)