import numpy as np
from math import *
import config_double_pendulum as conf
import config_admittance_shaping as admshaping

# Controller: # choose: pd | pdff | pdg | id | acc | imp | kDC | Zf
ctrl_type = 'Zf'
Kp = np.eye(conf.Model.nv) * 380.0
Kd = np.eye(conf.Model.nv) * 35.0

if ctrl_type == 'id':
    Kp[1, 1] *= 0.25
    Kd[1, 1] *= 0.30
# print(Kp)
# acceleration-based controller: PI(acc) -> PD(vel)
velKp = np.array([[60, 0], [0, 40]])
velKd = np.eye(conf.Model.nv) * 0.20

# Impedance Controller:
des_inertia = np.array([[0.2, 0], [0, 0.2]])
des_damping = np.array([[30., 0], [0, 90]])
des_stiffness = np.array([[60, 0], [0, 180]])

q_rlx = np.array([pi, .0])

# 2ndOrd Low-Pass-Filter Hf
of = admshaping.of
wf = admshaping.wf
ar = admshaping.ar
hk = np.zeros(conf.Model.nv)
hk1 = np.zeros(conf.Model.nv)
hk2 = np.zeros(conf.Model.nv)


def hf_transferfunct(new, act, last, llast):
    Dt = conf.dt
    dhk = (act - last)/Dt
    ddhk = (act - 2*last + llast)/(Dt**2)
    hk = (1 - ddhk/ar - 2*of*dhk/ar)*new
    hk2 = last.copy()
    hk1 = hk.copy()
    return hk
# endof Hf


def robot_controller(robotData, DesiredStates, UserStates, RobotStates, tau_int):
    Mq = robotData.M
    hq = robotData.C
    grav = robotData.g

    q, dq, ddq = RobotStates[0], RobotStates[1], RobotStates[2]
    qh, dqh, ddqh = UserStates[0], UserStates[1], UserStates[2]
    q_des, dq_des, ddq_des = DesiredStates[0], DesiredStates[1], DesiredStates[2]

    if ctrl_type == 'Zf':  # Admittance Shaping Controller
        filtered = hf_transferfunct(admshaping.Zf_acc.dot(ddq), hk, hk1, hk2)
        return admshaping.k_DC.dot(q - q_rlx) - admshaping.Zf_acc.dot(ddq)
        # return admshaping.k_DC.dot(q - q_rlx) - filtered

    if ctrl_type == 'kDC':  # DC gain compensation from Admittance Shaping (remember: k_DC < 0)
        return admshaping.k_DC.dot(q - q_rlx) # compensa a posicao relaxada (pi, 0)

    if ctrl_type == 'Zcd':
        # return Mq.dot( admshaping.I_des_inv.dot( tau_int ) ) - tau_int
        # return Mq.dot( admshaping.I_des_inv.dot( admshaping.imp_kp.dot(qh - q) + tau_int ) ) - tau_int
        return Mq.dot(admshaping.I_des_inv.dot(admshaping.imp_kp.dot(qh - q) + admshaping.imp_kd.dot(dqh - dq) + tau_int)) - tau_int

    if ctrl_type == 'imp':  # Impedance Control
        inv_id = np.linalg.inv(des_inertia)
        return Mq.dot(inv_id.dot(des_stiffness.dot(qh - q) + des_damping.dot(dqh - dq) + tau_int)) - tau_int

    if ctrl_type == 'acc':  # Acceleration-based control
        return Mq.dot(ddq_des) + velKp.dot(dq_des - dq) + velKd.dot(ddq_des - ddq) + hq + grav

    if ctrl_type == 'id':  # Inverse Dynamics Control, data_sim.nle contains C + g
        return Mq.dot(ddq_des + Kp.dot(q_des - q) + Kd.dot(dq_des - dq)) + robotData.nle

    if ctrl_type == 'pdff':  # PD+Feedforward Control
        return Kp.dot(q_des - q) + Kd.dot(dq_des - dq) + np.multiply(np.diag(Mq), ddq_des)

    if ctrl_type == 'pdg':  # PD Control + Grav Compensation
        return Kp.dot(q_des - q) + Kd.dot(dq_des - dq) + grav

    if ctrl_type == 'pd':  # PD Control
        return Kp.dot(q_des - q) + Kd.dot(dq_des - dq)
    else:
        return np.zeros(conf.Model.nv)

# end of controller
