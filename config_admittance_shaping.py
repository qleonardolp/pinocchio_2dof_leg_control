import pinocchio as pin
import config_double_pendulum as conf
import matplotlib.pyplot as plt
from control.matlab import *
from math import *
import numpy as np

# Main Parameters
r1 = (1 + 2*conf.WeightRatio)/(1 + conf.WeightRatio)  #
r2 = r1  #
# Romega = np.array([[1.7, .0], [.0, 1.7]])
# Rpeaks = np.array([[7., .0], [.0, 7.]])
Romega = np.array([[1., .0], [.0, 1.]])
Rpeaks = np.array([[1., .0], [.0, 1.]])
Rdc    = np.array([[r1, .0], [.0, r2]])

# 2ndOrd Low-Pass-Filter Hf
of = 0.7
wf = 1.5
ar = of**2 + wf**2

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
I_des_inv = np.linalg.inv(I_des)
omg_des = np.diag(Romega * hum_omega_n)
zeta_des = np.sqrt( 0.5*np.eye(2) - 0.5*np.sqrt(np.eye(2) - 4*rho*rho) )
print(' Id: ' + str(np.diag(I_des_inv)))

# From the Integral Admittance definition, the desired Impedance Parameters are:
imp_kp = I_des * np.square(omg_des)
imp_kd = I_des * (2 * zeta_des * omg_des)
print(' Kd Z: ' + str(np.diag(imp_kd)))

# Equation (36): stiffness and gravity compensation gain
k_DC = hum_inertia * hum_omega_n*hum_omega_n * (np.linalg.inv(Rdc) - np.eye(2))
print('~Kp Z: ' + str(np.diag(imp_kp - k_DC)))

thigh_jw = [ -(hum_zetas*hum_omega_n)[0, 0], hum_omega_n[0,0]*sqrt(1 - hum_zetas[0, 0]) ]
shank_jw = [ -(hum_zetas*hum_omega_n)[1, 1], hum_omega_n[1,1]*sqrt(1 - hum_zetas[1, 1]) ]
poles = np.array([np.complex(thigh_jw[0], thigh_jw[1]), np.complex(shank_jw[0], shank_jw[1])])
print("Human Poles: " + str(poles))

thigh_jw = [ -(zeta_des*omg_des)[0, 0], omg_des[0]*sqrt(1 - zeta_des[0, 0]) ]
shank_jw = [ -(zeta_des*omg_des)[1, 1], omg_des[1]*sqrt(1 - zeta_des[1, 1]) ]
poles = np.array([np.complex(thigh_jw[0], thigh_jw[1]), np.complex(shank_jw[0], shank_jw[1])])
print("Desired Poles: " + str(poles))

# Obtain Ie (robot inertia about the y axis)
exoThighJyy = conf.Model.inertias[1].toDynamicParameters()[6]
exoShankJyy = conf.Model.inertias[2].toDynamicParameters()[6]
Ie = np.array([[exoThighJyy, .0], [.0, exoShankJyy]])

# loop gain for acceleration feedback. Eq (93),(94) without K_L_hec and Eq(52)!:
K_loop = np.eye(conf.Model.nv)
pole_d = np.complex(thigh_jw[0], thigh_jw[1])
cplx   = np.complex(pole_d.real + of, pole_d.imag + wf)
K_L_cj = cplx * cplx.conjugate()  # norm^2
cplx   = np.complex(pole_d.real + of, pole_d.imag - wf)
K_L = cplx * cplx.conjugate()  # norm^2
K_loop[0, 0] = sqrt((K_L*K_L_cj).real)

pole_d = np.complex(shank_jw[0], shank_jw[1])
cplx   = np.complex(pole_d.real + of, pole_d.imag + wf)
K_L_cj = cplx * cplx.conjugate()  # norm
cplx   = np.complex(pole_d.real + of, pole_d.imag - wf)
K_L = cplx * cplx.conjugate()  # norm
K_loop[1, 1] = sqrt((K_L*K_L_cj).real)

K_loop = K_loop / ar
print(K_loop)
# K_loop esta muito grande
# Limite teorico Zf_acc = Ie
Zf_acc = np.array([[0.445, .0],[.0, 0.30]])*Ie
# Zf_acc = K_loop*Ie

# Bode Plot
Kd = np.diag(imp_kp - k_DC)[0]
Bd = np.diag(imp_kd)[0]
Md = np.diag(I_des)[0]
imp1 = tf([Md, Bd, Kd], [1, 0])
Kd = np.diag(imp_kp - k_DC)[1]
Bd = np.diag(imp_kd)[1]
Md = np.diag(I_des)[1]
imp2 = tf([Md, Bd, Kd], [1, 0])
print('Z_1(s) =')
print(imp1)

plt.figure('Bode')
mag, phase, om = bode(imp1, imp2, logspace(-2, 2, 100), plot=True)
plt.show(block=False)

Hf = tf([ar],[1, 2*of, ar])
#print(Hf)
plt.figure('Hf Bode')
mag, phase, om = bode(Hf, logspace(-2, 2, 100), plot=True)
plt.show(block=False)