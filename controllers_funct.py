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
#des_inertia = np.array([[0.2, 0], [0, 0.2]])
#des_damping = np.array([[30., 0], [0, 90]])
#des_stiffness = np.array([[60, 0], [0, 180]])
# Admittance Shaping Controller
des_inertia = 0.23 * admshaping.I_des
des_damping = admshaping.imp_kd
des_stiffness = admshaping.imp_kp - admshaping.k_DC

q_rlx = np.array([pi, .0])


def robotController(robotData, DesiredStates, UserStates, RobotStates, tau_int):
    Mq = robotData.M
    hq = robotData.C
    grav = robotData.g

    q, dq, ddq = RobotStates[0], RobotStates[1], RobotStates[2]
    qh, dqh, ddqh = UserStates[0], UserStates[1], UserStates[2]
    q_des, dq_des, ddq_des = DesiredStates[0], DesiredStates[1], DesiredStates[2]

    if ctrl_type == 'Zf':
        # return Mq.dot( admshaping.I_des_inv.dot( -admshaping.k_DC.dot(qh - q) + tau_int ) ) - tau_int
        # return Mq.dot( admshaping.I_des_inv.dot( (admshaping.imp_kp - admshaping.k_DC).dot(qh - q) + tau_int ) ) - tau_int
        return Mq.dot(admshaping.I_des_inv.dot((admshaping.imp_kp - admshaping.k_DC).dot(qh - q) + admshaping.imp_kd.dot(dqh - dq) + tau_int)) - tau_int

    if ctrl_type == 'kDC':  # DC gain compensation from Admittance Shaping (remember: k_DC < 0)
        # return admshaping.k_DC.dot(q - q_rlx) # compensa a posicao relaxada (pi, 0)
        return admshaping.k_DC.dot(q - qh)  # compensa a posicao relativa ao usuario

    if ctrl_type == 'imp':  # Impedance Control
        inv_id = np.linalg.inv(des_inertia)
        return Mq.dot(inv_id.dot(des_stiffness.dot(qh - q) + des_damping.dot(dqh - dq) + tau_int)) - tau_int
        # return des_stiffness.dot(q_des - q) + des_damping.dot(dq_des - dq) + data_sim.nle

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
