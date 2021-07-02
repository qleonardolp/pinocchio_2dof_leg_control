import pinocchio as pin
import numpy as np
from math import *
import config_double_pendulum as conf

# Main Parameters
Romega = np.array([[1., .0], [.0, 1.]])
Rpeaks = np.array([[.7, .0], [.0, .5]])
Rdc    = np.array([[2.3, .0], [.0, 1.9]])

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

# From equations (17) - (19):
I_des = hum_inertia * np.linalg.inv(Rdc * Romega*Romega)
omg_des = np.diag(Romega * hum_omega_n)
zeta_des = np.sqrt( 0.5*np.eye(2) - 0.5*np.sqrt(np.eye(2) - 4*rho*rho) )

# From the Integral Admittance definition, the desired Impedance Parameters are:
imp_kp = I_des * np.square(omg_des)
imp_kd = I_des * (2 * zeta_des * omg_des)

# Equation (36): stiffness and gravity compensation gain
k_DC = hum_inertia * hum_omega_n*hum_omega_n * (np.linalg.inv(Rdc) - np.eye(2))
thigh_jw = [ -(zeta_des*omg_des)[0, 0], omg_des[0]*sqrt(1 - zeta_des[0, 0]) ]
shank_jw = [ -(zeta_des*omg_des)[1, 1], omg_des[1]*sqrt(1 - zeta_des[1, 1]) ]

poles = np.array([np.complex(thigh_jw[0], thigh_jw[1]), \
                  np.complex(shank_jw[0], shank_jw[1])])

print("Poles: " + str(poles))